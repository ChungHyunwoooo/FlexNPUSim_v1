#ifndef AXI_LOGGER_H
#define AXI_LOGGER_H

#include "spdlog/spdlog.h"
#include "spdlog/sinks/stdout_color_sinks.h"
#include "spdlog/sinks/basic_file_sink.h"
#include <memory>
#include <string>

/**
 * AXI Logger Wrapper
 * Provides consistent logging interface across AXI library
 *
 * Log Levels:
 * - TRACE: Detailed signal transitions, every transaction event
 * - DEBUG: Configuration, initialization, arbiter decisions
 * - INFO: High-level operations (transactions complete, bus events)
 * - WARN: Unexpected conditions, potential issues
 * - ERROR: Errors, protocol violations, failures
 */
class AXILogger {
private:
    static std::shared_ptr<spdlog::logger> console_logger;
    static std::shared_ptr<spdlog::logger> file_logger;
    static bool initialized;

    static void initialize_if_needed() {
        if (!initialized) {
            // Console logger with color
            console_logger = spdlog::stdout_color_mt("axi_console");
            console_logger->set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%^%l%$] [%n] %v");
            console_logger->set_level(spdlog::level::info);  // Default: INFO

            // File logger
            file_logger = spdlog::basic_logger_mt("axi_file", "logs/axi.log");
            file_logger->set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%l] [%n] %v");
            file_logger->set_level(spdlog::level::trace);  // Log everything to file

            initialized = true;
        }
    }

public:
    /**
     * Set console log level
     */
    static void set_level(spdlog::level::level_enum level) {
        initialize_if_needed();
        console_logger->set_level(level);
    }

    /**
     * Get console logger
     */
    static std::shared_ptr<spdlog::logger> get_console() {
        initialize_if_needed();
        return console_logger;
    }

    /**
     * Get file logger
     */
    static std::shared_ptr<spdlog::logger> get_file() {
        initialize_if_needed();
        return file_logger;
    }

    /**
     * Convenience logging functions
     */
    template<typename... Args>
    static void trace(const char* fmt, Args&&... args) {
        initialize_if_needed();
        console_logger->trace(fmt, std::forward<Args>(args)...);
        file_logger->trace(fmt, std::forward<Args>(args)...);
    }

    template<typename... Args>
    static void debug(const char* fmt, Args&&... args) {
        initialize_if_needed();
        console_logger->debug(fmt, std::forward<Args>(args)...);
        file_logger->debug(fmt, std::forward<Args>(args)...);
    }

    template<typename... Args>
    static void info(const char* fmt, Args&&... args) {
        initialize_if_needed();
        console_logger->info(fmt, std::forward<Args>(args)...);
        file_logger->info(fmt, std::forward<Args>(args)...);
    }

    template<typename... Args>
    static void warn(const char* fmt, Args&&... args) {
        initialize_if_needed();
        console_logger->warn(fmt, std::forward<Args>(args)...);
        file_logger->warn(fmt, std::forward<Args>(args)...);
    }

    template<typename... Args>
    static void error(const char* fmt, Args&&... args) {
        initialize_if_needed();
        console_logger->error(fmt, std::forward<Args>(args)...);
        file_logger->error(fmt, std::forward<Args>(args)...);
    }

    /**
     * Flush all loggers
     */
    static void flush() {
        if (initialized) {
            console_logger->flush();
            file_logger->flush();
        }
    }
};

// Static member initialization (C++17 inline)
inline std::shared_ptr<spdlog::logger> AXILogger::console_logger = nullptr;
inline std::shared_ptr<spdlog::logger> AXILogger::file_logger = nullptr;
inline bool AXILogger::initialized = false;

/**
 * Convenience macros for logging
 */
#define AXI_LOG_TRACE(...) AXILogger::trace(__VA_ARGS__)
#define AXI_LOG_DEBUG(...) AXILogger::debug(__VA_ARGS__)
#define AXI_LOG_INFO(...)  AXILogger::info(__VA_ARGS__)
#define AXI_LOG_WARN(...)  AXILogger::warn(__VA_ARGS__)
#define AXI_LOG_ERROR(...) AXILogger::error(__VA_ARGS__)

#endif // AXI_LOGGER_H
