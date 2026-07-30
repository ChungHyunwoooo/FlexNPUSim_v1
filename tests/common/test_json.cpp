/**
 * @file test_json.cpp
 * @brief Contract tests for common/json.h (JSONC + $include + field helpers).
 *
 * Contracts under test:
 *  - JSONC: // and block comments parse
 *  - $include: only a single-key {"$include": path} object is replaced;
 *    paths resolve relative to the including file; expansion is recursive
 *  - circular $include throws (names the chain) instead of overflowing
 *  - load_or_throw: unopenable file -> error naming the path
 *  - expect_field: missing vs type errors are distinguished
 *  - optional_field: falls back on both missing and type mismatch
 */

#include "common/json.h"

#include <unistd.h>

#include <cassert>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

namespace fs = std::filesystem;
using flexnpu_sim::json::Json;

static fs::path g_dir;

static void write_file(const fs::path& rel, const std::string& text) {
    const fs::path p = g_dir / rel;
    fs::create_directories(p.parent_path());
    std::ofstream(p) << text;
}

static bool throws_containing(const fs::path& file, const std::string& needle) {
    try {
        (void)flexnpu_sim::json::load_or_throw(file);
    } catch (const std::exception& e) {
        return std::string(e.what()).find(needle) != std::string::npos;
    }
    return false;
}

static void test_jsonc_comments() {
    write_file("c.json",
               "// line comment\n"
               "{ /* block */ \"a\": 1 // trailing\n}");
    auto j = flexnpu_sim::json::load_or_throw(g_dir / "c.json");
    assert(j.at("a").get<int>() == 1);
}

static void test_include_file_relative_recursive() {
    // root -> sub/mid.json -> leaf.json, where "leaf.json" must resolve
    // relative to sub/ (the including file), not the root.
    write_file("root.json", R"({ "x": { "$include": "sub/mid.json" } })");
    write_file("sub/mid.json", R"({ "y": { "$include": "leaf.json" } })");
    write_file("sub/leaf.json", R"({ "z": 42 })");
    auto j = flexnpu_sim::json::load_or_throw(g_dir / "root.json");
    assert(j.at("x").at("y").at("z").get<int>() == 42);
}

static void test_include_single_key_rule() {
    // An object holding $include PLUS other keys is NOT an include node.
    write_file("multi.json", R"({ "m": { "$include": "leaf.json", "n": 1 } })");
    write_file("leaf.json", R"({ "z": 1 })");
    auto j = flexnpu_sim::json::load_or_throw(g_dir / "multi.json");
    assert(j.at("m").contains("$include"));  // untouched
    assert(j.at("m").at("n").get<int>() == 1);
}

static void test_include_cycle_throws() {
    write_file("a.json", R"({ "next": { "$include": "b.json" } })");
    write_file("b.json", R"({ "next": { "$include": "a.json" } })");
    assert(throws_containing(g_dir / "a.json", "circular $include"));
}

static void test_load_errors() {
    assert(throws_containing(g_dir / "nope.json", "cannot open JSON"));
    write_file("bad.json", "{ broken");
    assert(throws_containing(g_dir / "bad.json", "JSON parse error"));
}

static void test_field_helpers() {
    Json j = Json::parse(R"({ "dram": { "tier": "dramsim3", "banks": 16 } })");
    assert(flexnpu_sim::json::expect_field<std::string>(j, "/dram/tier") ==
           "dramsim3");
    // missing -> "missing required field"
    bool missing = false;
    try {
        (void)flexnpu_sim::json::expect_field<int>(j, "/dram/absent");
    } catch (const std::exception& e) {
        missing = std::string(e.what()).find("missing required field") !=
                  std::string::npos;
    }
    assert(missing);
    // type mismatch -> "type error at"
    bool type_err = false;
    try {
        (void)flexnpu_sim::json::expect_field<int>(j, "/dram/tier");
    } catch (const std::exception& e) {
        type_err = std::string(e.what()).find("type error at") !=
                   std::string::npos;
    }
    assert(type_err);
    // optional_field: fallback on missing AND on type mismatch
    assert(flexnpu_sim::json::optional_field<int>(j, "/dram/absent", 7) == 7);
    assert(flexnpu_sim::json::optional_field<int>(j, "/dram/tier", 9) == 9);
    assert(flexnpu_sim::json::optional_field<int>(j, "/dram/banks", 0) == 16);
}

int main() {
    g_dir = fs::temp_directory_path() /
            ("flexnpu_json_test_" + std::to_string(::getpid()));
    fs::remove_all(g_dir);
    fs::create_directories(g_dir);

    test_jsonc_comments();
    test_include_file_relative_recursive();
    test_include_single_key_rule();
    test_include_cycle_throws();
    test_load_errors();
    test_field_helpers();

    fs::remove_all(g_dir);
    std::cout << "test_json: all contracts hold\n";
    return 0;
}
