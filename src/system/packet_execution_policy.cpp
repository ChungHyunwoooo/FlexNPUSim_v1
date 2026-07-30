/**
 * @file packet_execution_policy.cpp
 * @brief Domain policy for packet-internal function execution.
 */

#include "system/packet_execution_policy.h"

namespace flexnpu_sim {

PacketExecutionDecision PacketExecutionPolicy::decide(const LayerDescriptor& layer,
                                                      const FunctionExecutionMeta& meta) {
    PacketExecutionDecision d{};
    d.bypass_input_dma = meta.i1_from_packet;
    d.bypass_output_dma = meta.has_internal_consumer;
    d.effective_input_size = d.bypass_input_dma ? 0u : layer.input.size;
    d.effective_output_size = d.bypass_output_dma ? 0u : layer.output.size;
    return d;
}

} // namespace flexnpu_sim

