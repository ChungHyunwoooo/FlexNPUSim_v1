# tests/CMakeLists.txt — regenerated module by module from the src review.
add_compile_options(-UNDEBUG)   # keep asserts live in tests

# ============================================================================
# common
# ============================================================================
add_executable(test_types common/test_types.cpp)
target_link_libraries(test_types flexnpusim_core)
add_test(NAME test_types COMMAND test_types)

add_executable(test_json common/test_json.cpp)
target_link_libraries(test_json flexnpusim_core)
add_test(NAME test_json COMMAND test_json)

add_executable(test_tile_descriptor common/test_tile_descriptor.cpp)
target_link_libraries(test_tile_descriptor flexnpusim_core)
add_test(NAME test_tile_descriptor COMMAND test_tile_descriptor)

add_executable(test_config common/test_config.cpp)
target_link_libraries(test_config flexnpusim_core)
add_test(NAME test_config COMMAND test_config)

add_executable(test_debug_log common/test_debug_log.cpp)
target_link_libraries(test_debug_log flexnpusim_core)
add_test(NAME test_debug_log COMMAND test_debug_log)

# ============================================================================
# systemc/bus
# ============================================================================
add_executable(test_axi_arbiter systemc/bus/test_axi_arbiter.cpp)
target_link_libraries(test_axi_arbiter flexnpusim_core)
add_test(NAME test_axi_arbiter COMMAND test_axi_arbiter)

foreach(t test_axi_roundtrip test_axi_bus_routing test_axi_transport_caps)
    add_executable(${t} systemc/bus/${t}.cpp)
    target_include_directories(${t} PRIVATE
        ${CMAKE_SOURCE_DIR}/tests/systemc/bus ${SYSTEMC_INCLUDE})
    target_link_libraries(${t} flexnpusim_core ${SYSTEMC_LIB}
        spdlog::spdlog_header_only pthread)
    add_test(NAME ${t} COMMAND ${t})
    set_tests_properties(${t} PROPERTIES TIMEOUT 60)
endforeach()

# ============================================================================
# systemc/memory
# ============================================================================
add_executable(test_dram_tiers systemc/memory/test_dram_tiers.cpp)
target_link_libraries(test_dram_tiers flexnpusim_core)
add_test(NAME test_dram_tiers COMMAND test_dram_tiers)

add_executable(test_memory_slave_timing systemc/memory/test_memory_slave_timing.cpp)
target_include_directories(test_memory_slave_timing PRIVATE ${SYSTEMC_INCLUDE})
target_link_libraries(test_memory_slave_timing flexnpusim_core ${SYSTEMC_LIB}
    spdlog::spdlog_header_only pthread)
add_test(NAME test_memory_slave_timing COMMAND test_memory_slave_timing)
set_tests_properties(test_memory_slave_timing PROPERTIES TIMEOUT 60)

# ============================================================================
# systemc/dma
# ============================================================================
add_executable(test_dma_model systemc/dma/test_dma_model.cpp)
target_link_libraries(test_dma_model flexnpusim_core)
add_test(NAME test_dma_model COMMAND test_dma_model)

add_executable(test_dma_profile systemc/dma/test_dma_profile.cpp)
target_link_libraries(test_dma_profile flexnpusim_core)
add_test(NAME test_dma_profile COMMAND test_dma_profile)

# ============================================================================
# systemc/npu
# ============================================================================
add_executable(test_gb_state systemc/npu/test_gb_state.cpp)
target_link_libraries(test_gb_state flexnpusim_core)
add_test(NAME test_gb_state COMMAND test_gb_state)

add_executable(test_npu_global_buffer systemc/npu/test_npu_global_buffer.cpp)
target_link_libraries(test_npu_global_buffer flexnpusim_core)
add_test(NAME test_npu_global_buffer COMMAND test_npu_global_buffer)

# ============================================================================
# compiler
# ============================================================================
add_executable(test_network_modes compiler/test_network_modes.cpp)
target_link_libraries(test_network_modes flexnpusim_core)
add_test(NAME test_network_modes COMMAND test_network_modes)

# ============================================================================
# model
# ============================================================================
add_executable(test_latency_model model/test_latency_model.cpp)
target_link_libraries(test_latency_model flexnpusim_core)
add_test(NAME test_latency_model COMMAND test_latency_model)

if(OpenCV_FOUND)
    add_executable(test_function_model model/test_function_model.cpp)
    target_link_libraries(test_function_model flexnpusim_core pthread)
    add_test(NAME test_function_model COMMAND test_function_model)
endif()

add_executable(test_latency_model_chain model/test_latency_model_chain.cpp)
target_link_libraries(test_latency_model_chain flexnpusim_core)
add_test(NAME test_latency_model_chain COMMAND test_latency_model_chain)

add_executable(test_tile_search compiler/test_tile_search.cpp)
target_link_libraries(test_tile_search flexnpusim_core)
add_test(NAME test_tile_search COMMAND test_tile_search)
