/**
 * @file packet_execution_policy.h
 * @brief Domain policy for packet-internal function execution.
 */

#pragma once

#include "compiler/dnn_image/dnn_image_format.h"
#include "compiler/dnn_image/dnn_image_packet.h"
#include <cstdint>

namespace flexnpu_sim {

struct PacketExecutionDecision {
    bool bypass_input_dma = false;
    bool bypass_output_dma = false;
    uint32_t effective_input_size = 0;
    uint32_t effective_output_size = 0;
};

class PacketExecutionPolicy {
public:
    static PacketExecutionDecision decide(const LayerDescriptor& layer,
                                          const FunctionExecutionMeta& meta);
};

} // namespace flexnpu_sim

