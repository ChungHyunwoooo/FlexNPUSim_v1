/**
 * @file latency_model.cpp
 * @brief Operand-Arrival-Driven Latency Model implementation
 *
 * Each cycle, update() does:
 *   1. accept offered operands up to the input/weight port cap and buffer
 *      capacity (the un-accepted remainder is re-offered by the caller)
 *   2. compute the issuable output count from the accepted operands + apply the
 *      generation rule (n_min gate, drain)
 *   3. record the issue (buffered level = accepted − consumed(issued), so no
 *      separate debit is needed)
 *   4. apply pipeline delay (in-flight counter: issue -> retire phase shift only)
 *   5. accept completion-ready events into the pass-completion timing queue
 *   6. service timing tokens up to the output port cap
 *   7. classify the state
 *
 * The model works entirely in integer operand counts — no fractional ratio.
 * Consumption is derived from the cumulative issue count, so nothing drifts
 * (docs/verif/2026-07-06-model-attack.md F1). The un-accepted supply is never
 * dropped: update() reports consumed_input and the caller keeps
 * offered − consumed to re-offer next cycle (backpressure).
 */

#include "model/latency/latency_model.h"
#include "common/flexnpu_config.h"
#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <string>

namespace flexnpu_sim {

// ============================================================================
// LmHwParams::from_config — fill the LM's buffers and port caps from the config.
// Port caps default to the GB read/write port widths (byte->operand boundary);
// buffers come from pe_buffers / accumulator.
// ============================================================================

LmHwParams LmHwParams::from_config(const config::FlexNpuSimConfig& cfg,
                                   const std::string& function_type) {
    LmHwParams hw;
    uint32_t elem = cfg.hw.compute.element_size_bytes;
    if (elem == 0) elem = 1;

    const auto& gb = cfg.hw.buffers.global;

    // Port caps (byte->operand boundary): the GB read/write port widths (bytes/
    // cycle) / element size default the LM's input/weight/output port caps in
    // operands/cycle — the unit the model works in.
    if (gb.read_port_bytes_per_cycle > 0) {
        hw.max_input_ops_per_cycle  = gb.read_port_bytes_per_cycle / elem;
        hw.max_weight_ops_per_cycle = hw.max_input_ops_per_cycle;
    }
    if (gb.write_port_bytes_per_cycle > 0)
        hw.max_output_ops_per_cycle = gb.write_port_bytes_per_cycle / elem;

    // Accumulator scalar capacity → pass-completion timing-queue event slots.
    // This queue does not hold physical stage results; it projects the capacity
    // constraint into the latency model's pass-event domain.
    if (cfg.hw.buffers.accumulator.capacity_kb > 0)
        hw.output_buf_capacity = (cfg.hw.buffers.accumulator.capacity_kb * 1024) / elem;

    // PE-internal input/weight buffers — bound how much GB-delivered data the
    // compute side can hold at once. Forces backpressure on DMA when full and
    // stall on compute when empty (e.g. across multi-tile weight transitions).
    config::PeBufferCfg pe = cfg.hw.buffers.pe_buffers;   // default unit caps
    if (!function_type.empty()) {
        auto it = cfg.hw.buffers.pe_buffers_per_function.find(function_type);
        if (it != cfg.hw.buffers.pe_buffers_per_function.end()) pe = it->second;
    }
    if (pe.input_buffer_capacity_kb > 0)
        hw.input_buf_capacity  = (pe.input_buffer_capacity_kb  * 1024) / elem;
    if (pe.weight_buffer_capacity_kb > 0)
        hw.weight_buf_capacity = (pe.weight_buffer_capacity_kb * 1024) / elem;

    // lm↔gb wiring consistency (warn once per process): a data-bus width that
    // is not a whole multiple of the element size cannot carry whole operands,
    // and a PE buffer larger than the entire GB is physically impossible.
    static bool checked = false;
    if (!checked) {
        checked = true;
        if (gb.read_port_bytes_per_cycle && gb.read_port_bytes_per_cycle % elem)
            std::cerr << "[latency] warning: GB read_port_bytes_per_cycle ("
                      << gb.read_port_bytes_per_cycle
                      << ") is not a multiple of element_size_bytes (" << elem << ")\n";
        if (gb.write_port_bytes_per_cycle && gb.write_port_bytes_per_cycle % elem)
            std::cerr << "[latency] warning: GB write_port_bytes_per_cycle ("
                      << gb.write_port_bytes_per_cycle
                      << ") is not a multiple of element_size_bytes (" << elem << ")\n";
        if (pe.input_buffer_capacity_kb && gb.capacity_kb &&
            pe.input_buffer_capacity_kb > gb.capacity_kb)
            std::cerr << "[latency] warning: PE input buffer ("
                      << pe.input_buffer_capacity_kb << " KB) exceeds global buffer ("
                      << gb.capacity_kb << " KB)\n";
        if (pe.weight_buffer_capacity_kb && gb.capacity_kb &&
            pe.weight_buffer_capacity_kb > gb.capacity_kb)
            std::cerr << "[latency] warning: PE weight buffer ("
                      << pe.weight_buffer_capacity_kb << " KB) exceeds global buffer ("
                      << gb.capacity_kb << " KB)\n";
    }

    return hw;
}

// ============================================================================
// Constructor
// ============================================================================

LmModel::LmModel(const LmParams& params,
                 const LmHwParams& hw,
                 const std::string& name)
    : name_(name), params_(params), hw_(hw)
{
    // A finite input/weight buffer must be able to hold the initial prefetch,
    // else the start threshold can never accumulate → deadlock.
    if (hw_.input_buf_capacity && params_.prefetch_in > hw_.input_buf_capacity)
        throw std::invalid_argument(
            "LmModel: prefetch_in (" + std::to_string(params_.prefetch_in) +
            ") exceeds input buffer capacity (" +
            std::to_string(hw_.input_buf_capacity) + ")");
    if (hw_.weight_buf_capacity && params_.prefetch_wt > hw_.weight_buf_capacity)
        throw std::invalid_argument(
            "LmModel: prefetch_wt (" + std::to_string(params_.prefetch_wt) +
            ") exceeds weight buffer capacity (" +
            std::to_string(hw_.weight_buf_capacity) + ")");

    for (uint32_t i = 0; i < params_.issue_latency; i++) {
        pipeline_delay_queue_.push(0);
    }
}

// ============================================================================
// Buffered operands — derived from cumulative counts (no fractional ratio).
//   consumed = operands released by the issued outputs, via the exact integer
//              input:output rate: floor(issued * fetch / write). Monotonic, and
//              exactly reconciles (issued=write_output -> consumed=fetch).
//   buffered = accepted - consumed (currently buffered operands).
// ============================================================================

uint64_t LmModel::input_consumed() const {
    return params_.write_output
        ? (issued_outputs_ * params_.fetch_operand0) / params_.write_output : 0;
}
uint64_t LmModel::weight_consumed() const {
    return (params_.fetch_operand1 == 0 || params_.write_output == 0)
        ? 0 : (issued_outputs_ * params_.fetch_operand1) / params_.write_output;
}
uint32_t LmModel::buffered_input() const {
    const uint64_t c = input_consumed();
    return static_cast<uint32_t>(
        cumulative_input_fetches_ > c ? cumulative_input_fetches_ - c : 0);
}
uint32_t LmModel::buffered_weight() const {
    const uint64_t c = weight_consumed();
    return static_cast<uint32_t>(
        cumulative_weight_fetches_ > c ? cumulative_weight_fetches_ - c : 0);
}

// Initial prefetch gate: before the first issue, the model holds until the
// start threshold (prefetch_in/wt) worth of operands has accumulated — it
// models "a conv needs a kernel window, a pool a window, before the first
// output". It releases once issuing has begun OR all operands have arrived
// (the latter guards against a prefetch set larger than the layer's supply).
bool LmModel::prefetch_pending() const {
    if (issued_outputs_ > 0) return false;
    const bool all_in = cumulative_input_fetches_ >= params_.fetch_operand0;
    const bool all_wt = (params_.fetch_operand1 == 0) ||
                        cumulative_weight_fetches_ >= params_.fetch_operand1;
    const bool in_short = !all_in && cumulative_input_fetches_  < params_.prefetch_in;
    const bool wt_short = !all_wt && cumulative_weight_fetches_ < params_.prefetch_wt;
    return in_short || wt_short;
}

// ============================================================================
// reset — called at the start of a new layer
// ============================================================================

void LmModel::reset(const LmParams& params) {
    params_ = params;

    output_buf_level_ = 0;
    cumulative_input_fetches_  = 0;
    cumulative_weight_fetches_ = 0;
    issued_outputs_    = 0;
    generated_outputs_ = 0;
    retire_backlog_    = 0;
    cycle_count_       = 0;

    while (!pipeline_delay_queue_.empty()) pipeline_delay_queue_.pop();
    for (uint32_t i = 0; i < params_.issue_latency; i++) {
        pipeline_delay_queue_.push(0);
    }
}

// ============================================================================
// accept — operands accepted this cycle = offered, limited by (1) the port cap
// (max operands/cycle) and (2) the remaining buffer space (0 = unlimited for
// either). The caller re-offers whatever is not accepted; nothing is dropped.
// ============================================================================

uint32_t LmModel::accept(uint32_t offered, uint32_t max_ops_per_cycle,
                         uint32_t capacity, uint32_t buffered) const {
    if (max_ops_per_cycle)                   // (1) port cap (max operands/cycle)
        offered = std::min(offered, max_ops_per_cycle);
    if (capacity)                            // (2) remaining buffer space
        offered = std::min(offered, capacity > buffered ? capacity - buffered : 0u);
    return offered;
}

// ============================================================================
// update — execute one cycle
// ============================================================================

LmStepResult LmModel::update(uint32_t offered_input,
                             uint32_t offered_weight) {
    cycle_count_++;
    LmStepResult result{};

    // ---- Step 1: accept offered operands, limited by the per-cycle intake
    //             port cap and the remaining buffer space ----
    uint32_t accept_input  = accept(offered_input,  hw_.max_input_ops_per_cycle,
                                    hw_.input_buf_capacity,  buffered_input());
    uint32_t accept_weight = accept(offered_weight, hw_.max_weight_ops_per_cycle,
                                    hw_.weight_buf_capacity, buffered_weight());

    cumulative_input_fetches_  += accept_input;
    cumulative_weight_fetches_ += accept_weight;

    result.consumed_input  = accept_input;
    result.consumed_weight = accept_weight;

    // ---- Step 2: compute issuable amount + generation rule (issue-time gate) ----
    uint32_t estimated_outputs = estimate_outputs();
    uint32_t issue_outputs = apply_generation_rule(estimated_outputs);

    // ---- Step 3: record issue (supply consumption is derived, not debited) ----
    // The buffered level is accepted − consumed(issued_outputs_), so advancing
    // the issue counter implicitly debits supply at the exact integer input:
    // output rate — no separate fractional bookkeeping needed.
    issued_outputs_ += issue_outputs;

    // ---- Step 4: pipeline delay (flight: phase shift only) ----
    pipeline_delay_queue_.push(issue_outputs);
    uint32_t pipeline_delayed_outputs = pipeline_delay_queue_.front();
    pipeline_delay_queue_.pop();

    // ---- Step 5: enqueue into the pass-completion timing queue ----
    // Already-debited events are never dropped: when the queue is full, the
    // overflow carries in retire_backlog_ and waits until backpressure clears.
    uint32_t ready_retire = pipeline_delayed_outputs + retire_backlog_;
    uint32_t cycle_outputs = ready_retire;
    if (hw_.output_buf_capacity > 0 && ready_retire > 0) {
        uint32_t queue_remaining = (output_buf_level_ < hw_.output_buf_capacity)
            ? hw_.output_buf_capacity - output_buf_level_ : 0;
        cycle_outputs = std::min(ready_retire, queue_remaining);
    }
    retire_backlog_ = ready_retire - cycle_outputs;
    output_buf_level_ += cycle_outputs;

    // ---- Step 6: Timing-token service (legacy name: writeback) ----
    // Port-cap-limited only (the output_buf capacity was applied in Step 5).
    uint32_t writeback = accept(output_buf_level_, hw_.max_output_ops_per_cycle);
    output_buf_level_ -= writeback;

    // ---- Step 7: count retirements (supply was debited at issue in Step 3) ----
    generated_outputs_ += cycle_outputs;

    // Layer-end tail is applied by NpuController after packet completion;
    // latency model does not gate is_done() on it (the sim is typically
    // memory-bound, so in-loop waits here would not stretch the packet).

    // ---- Step 8: Result ----
    result.outputs           = cycle_outputs;
    result.writeback_outputs = writeback;
    result.state = classify_state(estimated_outputs,
                                   pipeline_delayed_outputs,
                                   cycle_outputs);

    return result;
}

// ============================================================================
// is_done — whether all output has been generated
// ============================================================================

bool LmModel::is_done() const {
    return generated_outputs_ >= params_.write_output;
}

// ============================================================================
// estimate_outputs
// Level x ratio for each queue = outputs that queue alone can make. The
// smaller of the two queues is the bottleneck; n_max caps it.
// ============================================================================

uint32_t LmModel::estimate_outputs() const {
    if (params_.write_output == 0) return 0;
    // Outputs the accepted operands can support so far, minus those already
    // issued: floor(cumulative_accepted * write_output / fetch) - issued. This
    // is mathematically identical to the old floor(level * ratio) but stays in
    // integer operand counts (no double ratio, no fractional level).
    auto issuable = [&](uint64_t cumulative, uint32_t fetch) -> uint64_t {
        if (fetch == 0) return 0;
        const uint64_t sup = (cumulative * params_.write_output) / fetch;
        return sup > issued_outputs_ ? sup - issued_outputs_ : 0;
    };
    const uint64_t from_input =
        issuable(cumulative_input_fetches_, params_.fetch_operand0);

    // Weightless layer (pooling, activation): decided by input alone
    if (params_.fetch_operand1 == 0)
        return static_cast<uint32_t>(std::min<uint64_t>(from_input, params_.n_max));

    const uint64_t from_weight =
        issuable(cumulative_weight_fetches_, params_.fetch_operand1);
    return static_cast<uint32_t>(std::min({from_input, from_weight,
                                           static_cast<uint64_t>(params_.n_max)}));
}

// ============================================================================
// apply_generation_rule — issue-time gate
//   prefetch gate      : before the first issue, hold until the initial
//                 prefetch (prefetch_in/wt) has accumulated (prefetch_pending()).
//   arrival in progress : issue only when estimate >= n_min, else wait.
//   arrival complete    : issue min(remaining, max) — by conservation (doc
//                 §7.5) remaining supply == remaining events, so it is not
//                 bound to the issuable-count floor (avoids a deadlock where
//                 double round-down leaves the last event never emitted;
//                 docs/verif/2026-07-06-model-attack.md F2).
// ============================================================================

uint32_t LmModel::apply_generation_rule(
    uint32_t estimated_outputs) const {
    if (prefetch_pending()) return 0;  // still accumulating the start threshold

    uint64_t remaining = (issued_outputs_ < params_.write_output)
        ? params_.write_output - issued_outputs_ : 0;
    uint32_t cap = static_cast<uint32_t>(
        std::min<uint64_t>(remaining, params_.n_max));

    // Output-side backpressure: the pass-completion (accumulator) queue holds at
    // most output_buf_capacity events. When the queue plus its retire backlog
    // is full, the PE cannot start new passes, so cap the issue by the remaining
    // headroom — symmetric to the input-buffer clamp in Step 1. With an unbounded
    // output port the queue drains every cycle, so this never bites.
    if (hw_.output_buf_capacity > 0) {
        uint32_t occupied = output_buf_level_ + retire_backlog_;
        uint32_t headroom = (occupied < hw_.output_buf_capacity)
            ? hw_.output_buf_capacity - occupied : 0;
        if (headroom < cap) cap = headroom;
    }

    bool all_input_fetched =
        cumulative_input_fetches_ >= params_.fetch_operand0;
    bool all_weight_fetched = (params_.fetch_operand1 == 0) ||
        (cumulative_weight_fetches_ >= params_.fetch_operand1);
    if (all_input_fetched && all_weight_fetched) {
        return cap;  // tail drain: no supply floor (mathematically equivalent, §7.5)
    }

    // Require estimated >= n_min before issuing (lower-bound gate); otherwise
    // hold this cycle and wait for more supply.
    if (estimated_outputs >= params_.n_min) {
        return std::min(estimated_outputs, cap);
    }
    return 0;  // supply short: do not issue this cycle, wait
}

// ============================================================================
// classify_state — discrete state of this cycle
// ============================================================================

LmState LmModel::classify_state(
    uint32_t /* estimated_outputs */,
    uint32_t pipeline_delayed_outputs,
    uint32_t cycle_outputs) const {

    if (generated_outputs_ >= params_.write_output)
        return LmState::Done;

    if (prefetch_pending())
        return LmState::Prefetch;

    if (cycle_count_ <= params_.issue_latency)
        return LmState::PipelineFill;

    if (pipeline_delayed_outputs == 0)
        return LmState::Starved;

    if (pipeline_delayed_outputs < params_.n_min) {
        bool all_input_fetched =
            cumulative_input_fetches_ >= params_.fetch_operand0;
        return all_input_fetched ? LmState::Drain : LmState::Underflow;
    }

    if (cycle_outputs > 0)
        return LmState::Active;

    return LmState::Starved;
}

} // namespace flexnpu_sim
