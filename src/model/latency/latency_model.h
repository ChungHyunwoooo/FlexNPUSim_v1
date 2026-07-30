/**
 * @file latency_model.h
 * @brief Operand-Arrival-Driven Latency Model (pure C++ class)
 *
 * Input : input/weight operand counts arriving each cycle.
 * Output: outputs produced each cycle + actual consumption + state.
 *
 * Design principle — this model deliberately has NO functional (value)
 * accuracy. It abstracts the RTL unit(s) purely by operand-count ratios and
 * never looks at tensor values. Value-dependent timing (e.g. real
 * activation-based zero-skip) enters only after being converted to a count
 * correction at preprocessing time (the function model computes the values and
 * the sparsity measurement is the sole value->count bridge). Functional
 * accuracy is the sole responsibility of model/function (OpenCV, PyTorch 1:1).
 *
 * A pure C++ class, not an SC_MODULE. The controller (SC_MODULE) calls
 * update() once per cycle.
 */

#pragma once

#include <cstdint>
#include <queue>
#include <string>

namespace flexnpu_sim {

namespace config { struct FlexNpuSimConfig; }

// ============================================================================
// Latency Parameters — per-layer workload params (read from the DNN image)
// ============================================================================

struct LmParams {
    // --- Total traffic ---
    uint32_t fetch_operand0  = 0;  // total input operands of the layer
    uint32_t fetch_operand1 = 0;  // total weight operands of the layer
    uint32_t write_output   = 0;  // pass-completion event budget the packet compute accumulates
                                  // = passes_per_output * output element count

    // --- Compute ---
    uint32_t issue_latency        = 0;  // pipeline cycles from issue to completion-ready
    uint32_t n_min = 1;  // positive pass-issue width floor per cycle
    uint32_t n_max = 1;  // pass-issue width ceiling per cycle

    // --- Generation-start condition ---
    uint32_t prefetch_in       = 1;  // GB-resident input gate before compute starts
    uint32_t prefetch_wt      = 1;  // GB-resident weight gate before compute starts
};

// ============================================================================
// Latency HW Params — per-NPU hardware constraints (do not change per layer)
// ============================================================================

struct LmHwParams {
    // All quantities are OPERAND COUNTS (not bytes). The byte<->operand
    // conversion (via element_size_bytes) happens only in from_config, at the
    // config boundary — the model itself never sees bytes.
    uint32_t input_buf_capacity      = 0;  // input buffer capacity (operands). 0 = unlimited
    uint32_t weight_buf_capacity     = 0;  // weight buffer capacity (operands). 0 = unlimited
    // Pass-completion timing queue capacity in event slots. The runtime derives
    // it from accumulator scalar capacity; it is not a physical stage-result count.
    uint32_t output_buf_capacity     = 0;  // timing-queue capacity (events). 0 = unlimited

    // Port caps — the max operands (or events) this LM's ports move per cycle.
    // A property of the LM's own ports, independent of what feeds it (the GB, or
    // an upstream stage in a chain — the cap applies at either). 0 = unlimited.
    uint32_t max_input_ops_per_cycle  = 0;  // input port  (operands/cycle)
    uint32_t max_weight_ops_per_cycle = 0;  // weight port (operands/cycle)
    uint32_t max_output_ops_per_cycle = 0;  // output/timing-token service (events/cycle)

    // Factory: fills the LM's buffers and port caps from the config. The port
    // caps default to the GB read/write port widths (the byte->operand boundary:
    // data-bus bytes / element_size). Given a function_type ("conv2d", ...) it
    // applies the per-unit buffer override in hw.buffers.pe_buffers — an LM is
    // 1:1 with a function descriptor and each unit may have a different internal
    // buffer (2026-07-10). It warns on any lm<->gb inconsistency found.
    static LmHwParams from_config(const config::FlexNpuSimConfig& cfg,
                                  const std::string& function_type = std::string());
};

// ============================================================================
// Latency Model State — discrete per-cycle state classification
// ============================================================================

enum class LmState : uint8_t {
    PipelineFill,    // pipeline warm-up after issue, before completion-ready
    Prefetch,        // before first issue: accumulating the initial prefetch
    Starved,         // buffer empty, estimated output = 0
    Underflow,       // 0 < issuable amount < n_min (issue held)
    Active,          // pass completion occurs
    Drain,           // all input arrived, flushing the remainder
    Done             // all output generated
};

// ============================================================================
// LmStepResult — result of one update() cycle
// ============================================================================

struct LmStepResult {
    uint32_t outputs           = 0;  // pass-completion events accepted into the timing queue
    uint32_t consumed_input    = 0;  // input operands actually accepted
    uint32_t consumed_weight   = 0;  // weight operands actually accepted
    uint32_t writeback_outputs = 0;  // completion tokens serviced out of the timing queue
    LmState  state             = LmState::PipelineFill;
};

// ============================================================================
// LmModel — operand-arrival latency model (handles 1- or 2-input units)
// ============================================================================

class LmModel {
public:
    explicit LmModel(const LmParams& params,
                     const LmHwParams& hw = {},
                     const std::string& name = "");

    void set_name(const std::string& name) { name_ = name; }
    const std::string& name() const { return name_; }

    /**
     * @brief Execute one cycle.
     *
     * offered_input/weight: operand counts the controller offers this cycle.
     * Returns: actual consumption, generated output, writeback output, state.
     */
    LmStepResult update(uint32_t offered_input, uint32_t offered_weight);

    bool is_done() const;
    void reset(const LmParams& params);

    // Statistics
    uint64_t get_total_cycles() const { return cycle_count_; }
    uint64_t get_generated_outputs() const { return generated_outputs_; }
    uint32_t get_buffered_input() const { return buffered_input(); }    // operands
    uint32_t get_buffered_weight() const { return buffered_weight(); }  // operands

private:
    std::string name_;

    LmParams params_;
    LmHwParams hw_;

    // Output timing-queue occupancy (events). Input/weight buffered counts are
    // NOT stored — they are derived from the cumulative counters below via the
    // exact integer input:output rate (operand counts only; no double ratio).
    uint32_t output_buf_level_   = 0;  // pass-completion timing tokens

    // cumulative counters (all operand/event counts)
    uint64_t cumulative_input_fetches_  = 0;
    uint64_t cumulative_weight_fetches_ = 0;
    uint64_t issued_outputs_            = 0;  // cumulative at issue
    uint64_t generated_outputs_         = 0;  // cumulative at retire (observation)
    uint64_t cycle_count_               = 0;

    // pipeline-delay queue
    std::queue<uint32_t> pipeline_delay_queue_;

    // Carry for events that could not retire because the timing queue was full
    // (already issue-debited — dropping them would break conservation, so they
    // wait until backpressure clears).
    uint32_t retire_backlog_ = 0;

    // internal computations
    uint64_t input_consumed()  const;   // operands consumed by issued outputs
    uint64_t weight_consumed() const;
    uint32_t buffered_input()  const;      // buffered operands = accepted − consumed
    uint32_t buffered_weight() const;
    bool     prefetch_pending() const;     // still accumulating the initial prefetch
    uint32_t estimate_outputs() const;
    uint32_t apply_generation_rule(uint32_t estimated_outputs) const;
    LmState classify_state(uint32_t estimated_outputs,
                           uint32_t pipeline_delayed_outputs,
                           uint32_t cycle_outputs) const;
    // Operands accepted this cycle: offered, limited by the port cap (max
    // operands/cycle) and (if set) the remaining buffer space. 0 = unlimited.
    uint32_t accept(uint32_t offered, uint32_t max_ops_per_cycle,
                    uint32_t capacity = 0, uint32_t buffered = 0) const;
};

} // namespace flexnpu_sim
