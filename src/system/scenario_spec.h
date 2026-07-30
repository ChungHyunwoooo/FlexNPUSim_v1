#pragma once

#include <string>

namespace flexnpu_sim::system {

bool resolve_scenario_to_paths(const std::string& scenario,
                               const std::string& model_root_hint,
                               std::string& hw_conf_path,
                               std::string& network_csv_path,
                               std::string& inferred_npu_name,
                               std::string& err);

}  // namespace flexnpu_sim::system
