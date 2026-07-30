/**
 * @file ufb.h
 * @brief Ufb — the NPU's unit function block (compute datapath).
 *
 * The UFB is the PE array's timing model: given operands, it reports how many
 * output passes complete per cycle. It owns the LmModel (and, for a multi-
 * function packet, the controller builds an LmChain over per-stage UFBs).
 *
 * The GB→PE feed loop — reading available operands from the global buffer and
 * streaming them into the UFB cycle by cycle — is the CONTROLLER's dataflow
 * sequencing (real HW: the controller drives the datapath), so it lives in
 * NpuController::run_layer_compute, which drives this `lm`. This keeps the UFB
 * a pure compute unit and the controller the sequencer.
 */

#pragma once

#include <memory>
#include <string>

#include "model/latency/latency_model.h"        // LmModel
#include "compiler/dnn_image/dnn_image_format.h"  // FuncType

namespace flexnpu_sim {

class Ufb {
public:
    // PE-array timing model for the layer currently being computed. Rebuilt/
    // reset per layer by the controller's compute path.
    std::unique_ptr<LmModel> lm;

    // Profiler tag "layer<idx>_<type>" for the LM (and each chain stage).
    std::string layer_profiler_name(uint32_t idx, FuncType type) const {
        std::string n = "layer" + std::to_string(idx);
        switch (type) {
            case FuncType::Conv2D:          n += "_conv2d"; break;
            case FuncType::DepthwiseConv2D: n += "_dwconv"; break;
            case FuncType::FullyConnected:  n += "_fc"; break;
            case FuncType::Pooling:         n += "_pool"; break;
            case FuncType::Activation:      n += "_act"; break;
            case FuncType::MatMul:          n += "_matmul"; break;
            case FuncType::ElementWise:     n += "_ewise"; break;
        }
        return n;
    }
};

}  // namespace flexnpu_sim
