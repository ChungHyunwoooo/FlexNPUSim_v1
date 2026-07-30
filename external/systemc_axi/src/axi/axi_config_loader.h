#ifndef AXI_CONFIG_LOADER_H
#define AXI_CONFIG_LOADER_H

#include <string>
#include <fstream>
#include <sstream>
#include <map>
#include <vector>
#include <stdexcept>
#include <iostream>
#include "axi_logger.h"

/**
 * AXI Configuration Data Structure
 * Loaded from YAML configuration files
 */
struct AXIConfigData {
    // Protocol info
    std::string protocol_name;
    std::string protocol_version;

    // Bit widths
    unsigned int id_width = 4;
    unsigned int addr_width = 32;
    unsigned int data_width = 32;
    unsigned int user_width = 0;
    unsigned int len_width = 8;
    unsigned int lock_width = 1;

    // Feature flags
    bool enable_wid = false;
    bool enable_qos = true;
    bool enable_region = true;

    // Timing
    int slave_write_delay = 10;
    int slave_read_delay = 10;
    int master_timeout = 1000;
    int arbitration_delay = 1;

    // Topology
    unsigned int num_masters = 2;
    unsigned int num_slaves = 2;

    // Arbitration
    std::string arbitration_policy = "round_robin";  // round_robin, lru, priority

    // Memory map entry
    struct MemoryRegion {
        std::string name;
        uint64_t base;
        uint64_t size;
        unsigned int slave_id;
    };
    std::vector<MemoryRegion> memory_map;
};

/**
 * Simple YAML Parser for AXI Configuration
 * Supports basic YAML syntax (key: value, lists)
 */
class AXIConfigLoader {
private:
    std::map<std::string, std::string> config_map;
    std::vector<AXIConfigData::MemoryRegion> memory_regions;

    // Trim whitespace
    std::string trim(const std::string& str) {
        size_t first = str.find_first_not_of(" \t\r\n");
        if (first == std::string::npos) return "";
        size_t last = str.find_last_not_of(" \t\r\n");
        return str.substr(first, last - first + 1);
    }

    // Parse hex or decimal number
    uint64_t parse_number(const std::string& str) {
        std::string trimmed = trim(str);
        if (trimmed.substr(0, 2) == "0x" || trimmed.substr(0, 2) == "0X") {
            return std::stoull(trimmed, nullptr, 16);
        }
        return std::stoull(trimmed);
    }

    // Parse boolean
    bool parse_bool(const std::string& str) {
        std::string trimmed = trim(str);
        return (trimmed == "true" || trimmed == "True" || trimmed == "TRUE" || trimmed == "1");
    }

    // Parse YAML file
    void parse_yaml(const std::string& filename) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            throw std::runtime_error("Failed to open config file: " + filename);
        }

        std::string line;
        std::string current_section;
        AXIConfigData::MemoryRegion current_region;
        bool in_memory_map = false;

        while (std::getline(file, line)) {
            // Skip comments and empty lines
            size_t comment_pos = line.find('#');
            if (comment_pos != std::string::npos) {
                line = line.substr(0, comment_pos);
            }
            line = trim(line);
            if (line.empty()) continue;

            // Check indentation level
            size_t indent = 0;
            for (char c : line) {
                if (c == ' ') indent++;
                else break;
            }

            // Parse key-value pairs
            size_t colon_pos = line.find(':');
            if (colon_pos != std::string::npos) {
                std::string key = trim(line.substr(0, colon_pos));
                std::string value = trim(line.substr(colon_pos + 1));

                if (indent == 0) {
                    current_section = key;
                    in_memory_map = (key == "memory_map");
                } else if (indent == 2) {
                    // Section-level key
                    if (in_memory_map && value.empty()) {
                        // New memory region
                        if (!current_region.name.empty()) {
                            memory_regions.push_back(current_region);
                        }
                        current_region = AXIConfigData::MemoryRegion();
                    } else {
                        config_map[current_section + "." + key] = value;
                    }
                } else if (indent == 4 && in_memory_map) {
                    // Memory map entry
                    if (key == "name") current_region.name = value;
                    else if (key == "base") current_region.base = parse_number(value);
                    else if (key == "size") current_region.size = parse_number(value);
                    else if (key == "slave_id") current_region.slave_id = parse_number(value);
                }
            }
        }

        // Add last memory region
        if (in_memory_map && !current_region.name.empty()) {
            memory_regions.push_back(current_region);
        }

        file.close();
    }

    // Get value with default
    template<typename T>
    T get_value(const std::string& key, T default_value) {
        if (config_map.find(key) == config_map.end()) {
            return default_value;
        }
        std::string value = config_map[key];
        std::istringstream iss(value);
        T result;
        iss >> result;
        return result;
    }

    std::string get_string(const std::string& key, const std::string& default_value = "") {
        if (config_map.find(key) == config_map.end()) {
            return default_value;
        }
        std::string value = config_map[key];
        // Remove quotes if present
        if (value.size() >= 2 && value.front() == '"' && value.back() == '"') {
            return value.substr(1, value.size() - 2);
        }
        return value;
    }

public:
    /**
     * Load AXI configuration from YAML file
     */
    AXIConfigData load(const std::string& filename) {
        config_map.clear();
        memory_regions.clear();

        parse_yaml(filename);

        AXIConfigData config;

        // Protocol
        config.protocol_name = get_string("protocol.name", "AXI4_Default");
        config.protocol_version = get_string("protocol.version", "AXI4");

        // Bit widths
        config.id_width = parse_number(get_string("bit_widths.id_width", "4"));
        config.addr_width = parse_number(get_string("bit_widths.addr_width", "32"));
        config.data_width = parse_number(get_string("bit_widths.data_width", "32"));
        config.user_width = parse_number(get_string("bit_widths.user_width", "0"));
        config.len_width = parse_number(get_string("bit_widths.len_width", "8"));
        config.lock_width = parse_number(get_string("bit_widths.lock_width", "1"));

        // Features
        config.enable_wid = parse_bool(get_string("features.enable_wid", "false"));
        config.enable_qos = parse_bool(get_string("features.enable_qos", "true"));
        config.enable_region = parse_bool(get_string("features.enable_region", "true"));

        // Timing
        config.slave_write_delay = parse_number(get_string("timing.slave_write_delay", "10"));
        config.slave_read_delay = parse_number(get_string("timing.slave_read_delay", "10"));
        config.master_timeout = parse_number(get_string("timing.master_timeout", "1000"));
        config.arbitration_delay = parse_number(get_string("timing.arbitration_delay", "1"));

        // Topology
        config.num_masters = parse_number(get_string("topology.num_masters", "2"));
        config.num_slaves = parse_number(get_string("topology.num_slaves", "2"));

        // Arbitration
        config.arbitration_policy = get_string("arbitration.policy", "round_robin");

        // Memory map
        config.memory_map = memory_regions;

        return config;
    }

    /**
     * Print configuration (for debugging)
     */
    void print_config(const AXIConfigData& config) {
        AXI_LOG_INFO("=== AXI Configuration ===");
        AXI_LOG_INFO("Protocol: {} ({})", config.protocol_name, config.protocol_version);
        AXI_LOG_INFO("");
        AXI_LOG_INFO("Bit Widths:");
        AXI_LOG_INFO("  ID: {}", config.id_width);
        AXI_LOG_INFO("  ADDR: {}", config.addr_width);
        AXI_LOG_INFO("  DATA: {}", config.data_width);
        AXI_LOG_INFO("");
        AXI_LOG_INFO("Features:");
        AXI_LOG_INFO("  WID: {}", config.enable_wid ? "enabled" : "disabled");
        AXI_LOG_INFO("  QoS: {}", config.enable_qos ? "enabled" : "disabled");
        AXI_LOG_INFO("");
        AXI_LOG_INFO("Arbitration:");
        AXI_LOG_INFO("  Policy: {}", config.arbitration_policy);
        AXI_LOG_INFO("");
        AXI_LOG_INFO("Memory Map ({} regions):", config.memory_map.size());
        for (const auto& region : config.memory_map) {
            AXI_LOG_INFO("  {}: 0x{:x} - 0x{:x} (slave {})",
                        region.name, region.base, region.base + region.size, region.slave_id);
        }
        AXI_LOG_INFO("=========================");
    }
};

#endif // AXI_CONFIG_LOADER_H
