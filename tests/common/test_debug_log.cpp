/**
 * @file test_debug_log.cpp
 * @brief Contract tests for common/debug_log.h — the project's single
 *        logging system (FLEXNPU_LOG / FLEXNPU_LOG_CHANNELS env gates).
 *
 * Contracts:
 *  - level parsing: off/error/warn/info/debug/trace, case-insensitive,
 *    unknown/empty -> Off
 *  - gate: enabled iff configured level >= call level AND channel passes
 *  - channel filter: substring semantics; empty filter enables everything
 *
 * The level/channel accessors cache on first use, so the environment is
 * fixed before the first FLEXNPU_LOG_ENABLED evaluation.
 */

#include <cstdlib>

#include "common/debug_log.h"

#include <cassert>
#include <iostream>

using flexnpu_sim::LogLevel;
using flexnpu_sim::detail::parse_level_from_env;

static void test_level_parsing() {
    setenv("FLEXNPU_LOG", "TRACE", 1);
    assert(parse_level_from_env() == LogLevel::Trace);  // case-insensitive
    setenv("FLEXNPU_LOG", "warn", 1);
    assert(parse_level_from_env() == LogLevel::Warn);
    setenv("FLEXNPU_LOG", "banana", 1);
    assert(parse_level_from_env() == LogLevel::Off);    // unknown -> Off
    unsetenv("FLEXNPU_LOG");
    assert(parse_level_from_env() == LogLevel::Off);    // unset -> Off
}

static void test_gate_and_channel_filter() {
    // Fix the cached configuration: debug level, dma+state channels only.
    setenv("FLEXNPU_LOG", "debug", 1);
    setenv("FLEXNPU_LOG_CHANNELS", "dma,state", 1);

    assert(FLEXNPU_LOG_ENABLED(Debug, dma));
    assert(FLEXNPU_LOG_ENABLED(Info, state));    // below configured level
    assert(!FLEXNPU_LOG_ENABLED(Trace, dma));    // above configured level
    assert(!FLEXNPU_LOG_ENABLED(Debug, compute)); // channel filtered out
    // Filter semantics: a channel passes iff its full name appears inside
    // the filter string — so "dma,state" does NOT enable "dma_eng".
    assert(!FLEXNPU_LOG_ENABLED(Debug, dma_eng));
}

int main() {
    test_level_parsing();
    test_gate_and_channel_filter();
    std::cout << "test_debug_log: all contracts hold\n";
    return 0;
}
