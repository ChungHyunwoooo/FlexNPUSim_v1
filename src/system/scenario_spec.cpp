#include "system/scenario_spec.h"

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <map>
#include <sstream>
#include <vector>

namespace fs = std::filesystem;

namespace flexnpu_sim::system {

namespace {

std::string trim_copy(const std::string& s) {
    size_t b = 0;
    while (b < s.size() && std::isspace(static_cast<unsigned char>(s[b]))) b++;
    size_t e = s.size();
    while (e > b && std::isspace(static_cast<unsigned char>(s[e - 1]))) e--;
    return s.substr(b, e - b);
}

bool parse_design_network(const std::string& scenario,
                          std::string& design,
                          std::string& network) {
    const auto raw = trim_copy(scenario);
    if (raw.empty()) return false;

    const std::map<std::string, std::string> aliases = {
        {"npu-system", "midap:mobilenetv1"},
        {"nvdla-mobilenetv2", "nvdla-large:mobilenetv2"},
        {"timing-comparison", "midap:mobilenetv1"},
    };

    std::string normalized = raw;
    auto it = aliases.find(raw);
    if (it != aliases.end()) normalized = it->second;

    auto split_by = [&](char sep) -> bool {
        const size_t p = normalized.find(sep);
        if (p == std::string::npos || p == 0 || p + 1 >= normalized.size()) return false;
        design = normalized.substr(0, p);
        network = normalized.substr(p + 1);
        return true;
    };

    if (split_by(':')) return true;
    if (split_by('/')) return true;

    const size_t p = normalized.find('_');
    if (p != std::string::npos && p > 0 && p + 1 < normalized.size()) {
        design = normalized.substr(0, p);
        network = normalized.substr(p + 1);
        return true;
    }
    return false;
}

std::vector<std::string> model_root_candidates(const std::string& hint) {
    std::vector<std::string> roots;
    if (!hint.empty()) roots.push_back(hint);
    if (fs::exists("model/hw/npu") && fs::exists("model/sw")) roots.push_back(".");

    std::vector<std::string> unique_roots;
    for (const auto& r : roots) {
        if (std::find(unique_roots.begin(), unique_roots.end(), r) == unique_roots.end()) {
            unique_roots.push_back(r);
        }
    }
    return unique_roots;
}

}  // namespace

bool resolve_scenario_to_paths(const std::string& scenario,
                               const std::string& model_root_hint,
                               std::string& hw_conf_path,
                               std::string& network_csv_path,
                               std::string& inferred_npu_name,
                               std::string& err) {
    std::string design;
    std::string network;
    if (!parse_design_network(scenario, design, network)) {
        err = "invalid scenario format: '" + scenario +
              "' (expected <design>:<network> or <design>/<network>)";
        return false;
    }

    const auto roots = model_root_candidates(model_root_hint);
    if (roots.empty()) {
        err = "cannot find model root (expected model/hw/npu/ and model/sw/ under the CWD)";
        return false;
    }

    static const char* kWorkloadClasses[] = {"cnn", "attention"};

    std::vector<std::string> tried;
    for (const auto& root : roots) {
        const fs::path hw = fs::path(root) / "model" / "hw" / "npu" / (design + ".json");
        tried.push_back(hw.string());
        if (!fs::exists(hw)) continue;

        for (const char* cls : kWorkloadClasses) {
            const fs::path net =
                fs::path(root) / "model" / "sw" / cls / design / (network + ".csv");
            tried.push_back(net.string());
            if (fs::exists(net)) {
                hw_conf_path = hw.string();
                network_csv_path = net.string();
                inferred_npu_name = design;
                return true;
            }
        }
    }

    std::ostringstream oss;
    oss << "cannot resolve scenario '" << scenario << "'. Tried:";
    for (const auto& p : tried) oss << "\n  - " << p;
    err = oss.str();
    return false;
}

}  // namespace flexnpu_sim::system
