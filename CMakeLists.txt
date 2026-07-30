# FlexNPUSim — root build script
cmake_minimum_required(VERSION 3.16)
project(FlexNPUSim LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_compile_options(-Wall -Wextra -Wpedantic)

include(FetchContent)

# ============================================================================
# Dependencies (auto-downloaded when no local checkout exists)
# ============================================================================

# --- SystemC (built from source) ---
set(CMAKE_CXX_STANDARD 17 CACHE STRING "" FORCE)
set(CMAKE_CXX_STANDARD_REQUIRED ON CACHE BOOL "" FORCE)
set(DISABLE_COPYRIGHT_MESSAGE ON CACHE BOOL "" FORCE)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)

if(EXISTS ${CMAKE_SOURCE_DIR}/external/systemc/CMakeLists.txt)
    message(STATUS "SystemC: using local (external/systemc)")
    add_subdirectory(external/systemc EXCLUDE_FROM_ALL)
else()
    message(STATUS "SystemC: downloading via FetchContent...")
    FetchContent_Declare(systemc
        GIT_REPOSITORY https://github.com/accellera-official/systemc.git
        GIT_TAG        2.3.4
        GIT_SHALLOW    TRUE
    )
    FetchContent_MakeAvailable(systemc)
endif()
set(SYSTEMC_LIB systemc)
if(EXISTS ${CMAKE_SOURCE_DIR}/external/systemc/src)
    set(SYSTEMC_INCLUDE ${CMAKE_SOURCE_DIR}/external/systemc/src)
else()
    set(SYSTEMC_INCLUDE ${FETCHCONTENT_BASE_DIR}/systemc-src/src)
endif()

# --- spdlog (header-only mode) ---
# Sole consumer: external/systemc_axi's axi_logger (legacy register-bus lib).
# Dies together with that library (docs/reference/processor/control-model.md).
if(EXISTS ${CMAKE_SOURCE_DIR}/external/spdlog/CMakeLists.txt)
    message(STATUS "spdlog: using local (external/spdlog)")
    add_subdirectory(external/spdlog EXCLUDE_FROM_ALL)
else()
    message(STATUS "spdlog: downloading via FetchContent...")
    FetchContent_Declare(spdlog
        GIT_REPOSITORY https://github.com/gabime/spdlog.git
        GIT_TAG        v1.17.0
        GIT_SHALLOW    TRUE
    )
    FetchContent_MakeAvailable(spdlog)
endif()

# --- OpenCV (function model; optional) ---
find_package(OpenCV COMPONENTS core dnn imgproc)
if(OpenCV_FOUND)
    message(STATUS "OpenCV: ${OpenCV_VERSION} at ${OpenCV_DIR}")
else()
    message(WARNING "OpenCV not found. Function model features disabled.")
endif()

# --- systemc_axi (header-only INTERFACE; legacy) ---
# Provides AXI_master_if / AXI_slave_if used by the register-bus path
# (npu_controller, sim main). It resolves `#include "systemc/bus/axi/..."`
# through src/. This target dies when the processor control model replaces
# the legacy register bus (see docs/reference/processor/control-model.md).
add_library(systemc_axi INTERFACE)
target_include_directories(systemc_axi INTERFACE
    ${CMAKE_SOURCE_DIR}/external/systemc_axi/src
    ${CMAKE_SOURCE_DIR}/src
)
target_link_libraries(systemc_axi INTERFACE
    spdlog::spdlog_header_only
)

# ============================================================================
# Subdirectories (guards let the tree build while modules migrate in)
# ============================================================================
if(EXISTS ${CMAKE_SOURCE_DIR}/src/CMakeLists.txt)
    add_subdirectory(src)   # libraries + simulator binary (build/sim/flexnpusim)
endif()
if(EXISTS ${CMAKE_SOURCE_DIR}/tools/CMakeLists.txt)
    add_subdirectory(tools)
endif()
if(EXISTS ${CMAKE_SOURCE_DIR}/tests/CMakeLists.txt)
    enable_testing()
    add_subdirectory(tests)
endif()
