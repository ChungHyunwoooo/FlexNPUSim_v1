/**
 * @file json.h
 * @brief Project-wide nlohmann::json wrapper with JSONC + helpers.
 *
 * - JSONC parsing (single-line // and block comments).
 * - $include directive: any object containing only {"$include": "path"} is
 *   replaced inline with the parsed contents of that file (recursive).
 * - load_or_throw, expect_field — concise loaders that produce structured
 *   error messages with file:line + JSON pointer.
 *
 * Usage:
 *   auto j = flexnpu_sim::json::load_or_throw("configs/experiment.json");
 *   auto tier = flexnpu_sim::json::expect_field<std::string>(j, "/dram/tier");
 */

#pragma once

#include <nlohmann/json.hpp>

#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace flexnpu_sim::json {

using Json = ::nlohmann::json;

// ============================================================================
// $include resolution
//
// Walk the JSON tree; replace every object that has exactly one key "$include"
// with the parsed contents of the referenced file. Paths are resolved relative
// to base_dir (the directory of the file currently being parsed).
// ============================================================================

inline Json parse_jsonc(const std::string& text, const std::string& source_label) {
    try {
        // allow_exceptions=true, ignore_comments=true.
        return Json::parse(text, /*cb=*/nullptr, /*allow_exceptions=*/true,
                           /*ignore_comments=*/true);
    } catch (const Json::parse_error& e) {
        std::ostringstream oss;
        oss << "JSON parse error in " << source_label
            << " at byte " << e.byte << ": " << e.what();
        throw std::runtime_error(oss.str());
    }
}

namespace detail {

inline Json load_with_stack(const std::filesystem::path& path,
                            std::vector<std::filesystem::path>& stack);

// Circular $include is an error, not a hang: the include stack of files
// currently being expanded is threaded through the recursion and a revisit
// throws with the full chain.
inline void resolve_includes_inplace(Json& node,
                                     const std::filesystem::path& base_dir,
                                     std::vector<std::filesystem::path>& stack) {
    if (node.is_object()) {
        if (node.size() == 1 && node.contains("$include")) {
            const auto& inc = node.at("$include");
            if (!inc.is_string()) {
                throw std::runtime_error("$include must be a string path");
            }
            std::filesystem::path target = inc.get<std::string>();
            if (target.is_relative()) target = base_dir / target;
            node = load_with_stack(target, stack);
            return;  // node fully replaced; nothing more to recurse here.
        }
        for (auto& kv : node.items()) {
            resolve_includes_inplace(kv.value(), base_dir, stack);
        }
    } else if (node.is_array()) {
        for (auto& el : node) {
            resolve_includes_inplace(el, base_dir, stack);
        }
    }
}

inline Json load_with_stack(const std::filesystem::path& path,
                            std::vector<std::filesystem::path>& stack) {
    const auto canon = std::filesystem::weakly_canonical(path);
    for (const auto& open : stack) {
        if (open == canon) {
            std::ostringstream oss;
            oss << "circular $include: ";
            for (const auto& p : stack) oss << p.string() << " -> ";
            oss << canon.string();
            throw std::runtime_error(oss.str());
        }
    }
    std::ifstream ifs(path);
    if (!ifs.is_open()) {
        throw std::runtime_error("cannot open JSON: " + path.string());
    }
    std::ostringstream buf;
    buf << ifs.rdbuf();
    Json j = parse_jsonc(buf.str(), path.string());
    stack.push_back(canon);
    resolve_includes_inplace(j, path.parent_path(), stack);
    stack.pop_back();
    return j;
}

}  // namespace detail

inline void resolve_includes_inplace(Json& node,
                                     const std::filesystem::path& base_dir) {
    std::vector<std::filesystem::path> stack;
    detail::resolve_includes_inplace(node, base_dir, stack);
}

inline Json load_or_throw(const std::filesystem::path& path) {
    std::vector<std::filesystem::path> stack;
    return detail::load_with_stack(path, stack);
}

// ============================================================================
// expect_field — fetch by JSON pointer with structured error.
// ============================================================================

template <typename T>
T expect_field(const Json& root, const std::string& json_pointer) {
    try {
        const Json& v = root.at(::nlohmann::json_pointer<Json>(json_pointer));
        return v.get<T>();
    } catch (const Json::out_of_range&) {
        throw std::runtime_error("missing required field: " + json_pointer);
    } catch (const Json::type_error& e) {
        throw std::runtime_error("type error at " + json_pointer + ": " + e.what());
    }
}

template <typename T>
T optional_field(const Json& root, const std::string& json_pointer, T fallback) {
    try {
        const Json& v = root.at(::nlohmann::json_pointer<Json>(json_pointer));
        return v.get<T>();
    } catch (const Json::out_of_range&) {
        return fallback;
    } catch (const Json::type_error&) {
        return fallback;
    }
}

}  // namespace flexnpu_sim::json
