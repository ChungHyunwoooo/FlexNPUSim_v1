/**
 * @file config_loader.h
 * @brief Single entry point for loading user-facing JSON config (R0c).
 */

#pragma once

#include "common/flexnpu_config.h"

#include <filesystem>

namespace flexnpu_sim::config {

/**
 * @brief Load and validate a FlexNpuSimConfig from a JSONC file.
 *
 * Resolves $include directives recursively (relative to the file's directory).
 * Throws std::runtime_error on parse / type / missing-required-field failures
 * with a structured message identifying file + JSON pointer.
 */
/// @param require_network_source  When true (default), parse_network throws
///        if neither csv nor json source is given. Set to false when loading
///        HW-only JSONs (hw.json files in model/hw/npu/) that don't carry a
///        workload section.
FlexNpuSimConfig load_flexnpu_config(const std::filesystem::path& path,
                                      bool require_network_source = true);

}  // namespace flexnpu_sim::config
