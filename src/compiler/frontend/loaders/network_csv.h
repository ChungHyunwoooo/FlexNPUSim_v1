#pragma once

#include <string>

#include "common/flexnpu_config.h"
#include "compiler/dnn_image/dnn_image_generator.h"

namespace flexnpu_sim::frontend {

struct UnifiedCsvModel {
    DnnImage image{};
    NetworkConfig net_cfg{};
    uint32_t num_layers = 0;
    uint32_t num_layer_packets = 0;
};

bool generate_network_from_layer_spec_csv(const std::string& path,
                                          const config::FlexNpuSimConfig& cfg,
                                          UnifiedCsvModel& out,
                                          std::string& err);

}  // namespace flexnpu_sim::frontend

