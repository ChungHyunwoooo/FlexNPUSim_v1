/**
 * @file latency_model_chain.h
 * @brief Cycle-streaming chain of LMs (one stage per function descriptor). A
 *        stage's serviced output operands feed the next stage each cycle.
 *
 * Stage k is offered, every cycle, the accumulated output operands of stage k-1
 * that it has not yet accepted (the carry). Nothing is ever dropped: when a
 * stage's input buffer (B^I) back-pressures, the operands wait in the carry —
 * so conservation holds through the chain even with finite per-stage buffers.
 *
 * Forwarding is same-cycle (combinational hand-off): each stage's own
 * pipeline depth l already models its latency, so the chain adds no
 * artificial inter-stage registers. Weights are offered per stage by the
 * caller (typically the whole tensor once, on the first cycle).
 */

#pragma once

#include <cstdint>
#include <functional>
#include <vector>

#include "model/latency/latency_model.h"

namespace flexnpu_sim {

class LmChain {
public:
    /// Non-owning: stages must outlive the chain. The caller (npu_controller)
    /// owns the topology — which stage feeds which stage's input/weight is read
    /// from the DNN-image operand_src there, not assumed here.
    explicit LmChain(std::vector<LmModel*> stages)
        : stages_(std::move(stages)), carry_(stages_.size(), 0) {}

    bool empty() const { return stages_.empty(); }
    size_t size() const { return stages_.size(); }

    /// One cycle. `upstream` = output operands offered by the producer in front
    /// of the chain this cycle. `offer_weight(d)` returns the weight operands
    /// offered to stage d this cycle. Returns the final stage's serviced
    /// output-operand count this cycle.
    uint32_t step(uint32_t upstream,
                  const std::function<uint32_t(size_t)>& offer_weight,
                  const std::function<void(size_t, const LmStepResult&)>&
                      on_stage = nullptr) {
        if (stages_.empty()) return upstream;
        carry_[0] += upstream;
        uint32_t final_writeback = 0;
        for (size_t d = 0; d < stages_.size(); ++d) {
            const uint32_t w = offer_weight ? offer_weight(d) : 0;
            const LmStepResult r = stages_[d]->update(carry_[d], w);
            carry_[d] -= r.consumed_input;   // unaccepted operands wait here
            if (d + 1 < stages_.size()) {
                carry_[d + 1] += r.writeback_outputs;
            } else {
                final_writeback = r.writeback_outputs;
            }
            if (on_stage) on_stage(d, r);
        }
        return final_writeback;
    }

    bool done() const {
        for (const auto* s : stages_)
            if (!s->is_done()) return false;
        return true;
    }

    /// Operands parked between stages (diagnostics; 0 when drained).
    uint64_t carry_total() const {
        uint64_t t = 0;
        for (uint32_t c : carry_) t += c;
        return t;
    }

private:
    std::vector<LmModel*> stages_;
    std::vector<uint32_t> carry_;
};

}  // namespace flexnpu_sim
