# src/CMakeLists.txt — FlexNPUSim libraries and simulator binary
#
# flexnpusim_core    : pure C++, no SystemC. The split is by file selection,
#                      not by directory — systemc/*/model/*.cpp are behavioral
#                      models compiled into core; only signal-level modules
#                      need SystemC.
# flexnpusim_systemc : SystemC modules; links core + systemc_axi + SystemC.
#
# Include style: one style only — full src-relative paths
# (#include "systemc/npu/npu_controller.h"); src/ is the single include root.

# ============================================================================
# flexnpusim_core (pure C++)
# ============================================================================
set(CORE_SOURCES
    # common — config, types, tile descriptor
    common/config_loader.cpp
    common/tile_descriptor.cpp
    # model — event-based latency model
    model/latency/latency_model.cpp
    # compiler — network -> dnn image (descriptor stream)
    compiler/dnn_image/dnn_image_compiler.cpp
    compiler/dnn_image/tile_search.cpp
    compiler/dnn_image/dnn_image_packet.cpp
    compiler/dnn_image/dnn_image_generator.cpp
    # compiler — unified network CSV loader (the production input path)
    compiler/frontend/loaders/network_csv.cpp
    # zero-skipping preprocessing (sparsity patch; body is OpenCV-guarded)
    model/function/sparsity_preprocess.cpp
    # systemc component behavioral models (pure C++; wrappers are SystemC)
    systemc/memory/model/layer_timing_strategy.cpp
    systemc/memory/dram/ideal/dram_ideal.cpp
    systemc/memory/dram/bandwidth/dram_bandwidth.cpp
    systemc/memory/dram/bank/bank_address.cpp
    systemc/memory/dram/bank/dram_bank.cpp
    systemc/memory/dram/cycle/dram_cycle.cpp
    systemc/memory/dram/factory.cpp
    systemc/npu/model/npu_global_buffer.cpp
    systemc/dma/model/dma_model.cpp
    # system — packet execution policy (the NPU controller's issue rules)
    system/packet_execution_policy.cpp
)

# model/function — OpenCV-backed functional model (PyTorch 1:1)
if(OpenCV_FOUND)
    list(APPEND CORE_SOURCES
        model/function/function_model.cpp
        model/function/function_model_utils.cpp
        model/function/conv2d.cpp
        model/function/depthwise_conv2d.cpp
        model/function/pooling.cpp
        model/function/fully_connected.cpp
        model/function/activation.cpp
        model/function/matmul.cpp
        model/function/elementwise.cpp
        model/function/sparsity_profiler.cpp
        model/function/ofm_file_generator.cpp
    )
endif()

add_library(flexnpusim_core ${CORE_SOURCES})

target_include_directories(flexnpusim_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/external
)

if(OpenCV_FOUND)
    target_link_libraries(flexnpusim_core PUBLIC ${OpenCV_LIBS})
    target_compile_definitions(flexnpusim_core PUBLIC HAVE_OPENCV)
endif()

# --- DRAMSim3 (default DRAM backend) ---
include(FetchContent)
if(EXISTS ${CMAKE_SOURCE_DIR}/external/dramsim3/CMakeLists.txt)
    message(STATUS "DRAMSim3: using local (external/dramsim3)")
    add_subdirectory(${CMAKE_SOURCE_DIR}/external/dramsim3 dramsim3-build EXCLUDE_FROM_ALL)
    set(DRAMSIM3_SRC_DIR ${CMAKE_SOURCE_DIR}/external/dramsim3)
else()
    message(STATUS "DRAMSim3: downloading via FetchContent (umd-memsys/DRAMsim3 @ 1.0.0)...")
    FetchContent_Declare(dramsim3
        GIT_REPOSITORY https://github.com/umd-memsys/DRAMsim3.git
        GIT_TAG        1.0.0
        GIT_SHALLOW    TRUE
    )
    FetchContent_MakeAvailable(dramsim3)
    set(DRAMSIM3_SRC_DIR ${dramsim3_SOURCE_DIR})
endif()
target_compile_definitions(flexnpusim_core PUBLIC HAVE_DRAMSIM3
    DRAMSIM3_CONFIG_DIR="${DRAMSIM3_SRC_DIR}/configs")
target_include_directories(flexnpusim_core PUBLIC ${DRAMSIM3_SRC_DIR}/src)
target_link_libraries(flexnpusim_core PUBLIC dramsim3)

# ============================================================================
# flexnpusim_systemc (SystemC modules)
# ============================================================================
# AXI signal width is a compiled MAXIMUM; the runtime bus width (config
# axi.data_width_bits) selects beat accounting up to this ceiling. 1024b is
# the AXI4 maximum data width; no value above 1024 is valid.
set(FLEXNPUSIM_AXI_MAX_DATA_W 1024 CACHE STRING
    "Compiled AXI signal width (runtime bus width must be <= this; AXI4 max 1024)")

set(SYSTEMC_SOURCES
    systemc/npu/npu_controller.cpp
    systemc/npu/npu_controller_timeline.cpp
    systemc/npu/npu_controller_regs.cpp
    systemc/npu/npu_controller_dma.cpp
    systemc/npu/npu_controller_read.cpp
    systemc/npu/npu_controller_tile.cpp
    systemc/npu/npu_controller_compute.cpp
    systemc/npu/npu_controller_write.cpp
    system/flexnpusim_system.cpp
    system/performance_report.cpp
    system/scenario_spec.cpp
)

add_library(flexnpusim_systemc STATIC ${SYSTEMC_SOURCES})
target_include_directories(flexnpusim_systemc PUBLIC ${SYSTEMC_INCLUDE})
target_compile_definitions(flexnpusim_systemc PUBLIC
    FLEXNPUSIM_DMA_AXI_DATA_W=${FLEXNPUSIM_AXI_MAX_DATA_W})
target_link_libraries(flexnpusim_systemc PUBLIC
    flexnpusim_core
    systemc_axi
    ${SYSTEMC_LIB}
    pthread
)

# ============================================================================
# Simulator binaries -> build/sim/
# ============================================================================
set(FLEXNPUSIM_SIM_RUNTIME_DIR "${CMAKE_BINARY_DIR}/sim")

# The production entrypoint: SystemC full-system simulator.
add_executable(flexnpusim flexnpusim_main.cpp)
set_target_properties(flexnpusim PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${FLEXNPUSIM_SIM_RUNTIME_DIR}")
target_link_libraries(flexnpusim
    flexnpusim_systemc
    systemc_axi
    ${SYSTEMC_LIB}
    pthread)
