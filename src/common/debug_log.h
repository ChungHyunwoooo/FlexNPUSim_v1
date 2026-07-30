#pragma once

// Lightweight env-var-gated logger.
//
// Activate via FLEXNPU_LOG={off,error,warn,info,debug,trace}; default = off.
// Zero-overhead when level is off (single int compare, no formatting).
//
// Channels (FLEXNPU_LOG_CHANNELS=dma,state,desc,read,compute,write,gb)
// allow filtering by subsystem; empty = all channels enabled.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <strings.h>  // strcasecmp (POSIX)

namespace flexnpu_sim {

enum class LogLevel : int {
    Off   = 0,
    Error = 1,
    Warn  = 2,
    Info  = 3,
    Debug = 4,
    Trace = 5,
};

namespace detail {

inline LogLevel parse_level_from_env() {
    const char* s = std::getenv("FLEXNPU_LOG");
    if (!s || !*s) return LogLevel::Off;
    if (!::strcasecmp(s,"off"))   return LogLevel::Off;
    if (!::strcasecmp(s,"error")) return LogLevel::Error;
    if (!::strcasecmp(s,"warn"))  return LogLevel::Warn;
    if (!::strcasecmp(s,"info"))  return LogLevel::Info;
    if (!::strcasecmp(s,"debug")) return LogLevel::Debug;
    if (!::strcasecmp(s,"trace")) return LogLevel::Trace;
    return LogLevel::Off;
}

inline std::string parse_channels_from_env() {
    const char* s = std::getenv("FLEXNPU_LOG_CHANNELS");
    return s ? std::string(s) : std::string();
}

inline LogLevel& level() {
    static LogLevel lv = parse_level_from_env();
    return lv;
}

inline const std::string& channels() {
    static std::string ch = parse_channels_from_env();
    return ch;
}

inline bool channel_enabled(const char* ch) {
    const std::string& filter = channels();
    if (filter.empty()) return true;
    // A channel passes iff its full name appears inside the filter string
    // ("dma,state" enables dma and state; it does NOT enable dma_eng).
    return filter.find(ch) != std::string::npos;
}

}  // namespace detail

}  // namespace flexnpu_sim

#define FLEXNPU_LOG_ENABLED(lv, ch)                                            \
    (static_cast<int>(::flexnpu_sim::detail::level()) >=                       \
         static_cast<int>(::flexnpu_sim::LogLevel::lv) &&                      \
     ::flexnpu_sim::detail::channel_enabled(#ch))

#define FLEXNPU_LOG(lv, ch, ...)                                               \
    do {                                                                       \
        if (FLEXNPU_LOG_ENABLED(lv, ch)) {                                     \
            std::fprintf(stderr, "[" #ch "] " __VA_ARGS__);                    \
            std::fputc('\n', stderr);                                          \
        }                                                                      \
    } while (0)
