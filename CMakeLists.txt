cmake_minimum_required(VERSION 3.31)
project(GANSU LANGUAGES CXX)

# --- C++ / SYCL 設定 ---
set(CMAKE_CXX_STANDARD 17)

# --- SYCL compile flags for CUDA ---
set(SYCL_COMMON_COMPILE
    -fsycl
    -fsycl-targets=nvptx64-nvidia-cuda
    -fsycl-device-code-split=per_kernel
    -g -O0
)

set(SYCL_CUDA_BACKEND_OPTS
    -Xsycl-target-backend=nvptx64-nvidia-cuda "--cuda-gpu-arch=sm_86"
)

# --- Include directories ---
include_directories(
    include_sycl
    /home/mikei/docker_env/oneMath/include
    /home/mikei/docker_env/oneMath/build/bin
    /home/mikei/docker_env/oneMath/examples/include
    /home/mikei/docker_env/oneDPL/include
)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${SYCL_FLAGS}")

# --- 静的ライブラリ ---
add_library(int1e STATIC src_sycl/int1e.dp.cpp)
add_library(int2e STATIC src_sycl/int2e.dp.cpp)
add_library(int2c2e STATIC src_sycl/int2c2e.dp.cpp)
add_library(int3c2e STATIC src_sycl/int3c2e.dp.cpp)
add_library(int2e_direct STATIC src_sycl/int2e_direct.dp.cpp)

add_library(grad_w STATIC src_sycl/grad/grad_w.dp.cpp)
add_library(grad_s STATIC src_sycl/grad/grad_s.dp.cpp)
add_library(grad_k STATIC src_sycl/grad/grad_k.dp.cpp)
add_library(grad_v STATIC src_sycl/grad/grad_v.dp.cpp)
add_library(grad_g STATIC src_sycl/grad/grad_g.dp.cpp)


# Apply SYCL flags to static libraries
target_compile_options(int1e PRIVATE ${SYCL_COMMON_COMPILE} ${SYCL_CUDA_BACKEND_OPTS})
target_compile_options(int2e PRIVATE ${SYCL_COMMON_COMPILE} ${SYCL_CUDA_BACKEND_OPTS})
target_compile_options(int2c2e PRIVATE ${SYCL_COMMON_COMPILE} ${SYCL_CUDA_BACKEND_OPTS})
target_compile_options(int3c2e PRIVATE ${SYCL_COMMON_COMPILE} ${SYCL_CUDA_BACKEND_OPTS})
target_compile_options(int2e_direct PRIVATE ${SYCL_COMMON_COMPILE} ${SYCL_CUDA_BACKEND_OPTS})

target_compile_options(grad_w PRIVATE ${SYCL_COMMON_COMPILE} ${SYCL_CUDA_BACKEND_OPTS})
target_compile_options(grad_s PRIVATE ${SYCL_COMMON_COMPILE} ${SYCL_CUDA_BACKEND_OPTS})
target_compile_options(grad_k PRIVATE ${SYCL_COMMON_COMPILE} ${SYCL_CUDA_BACKEND_OPTS})
target_compile_options(grad_v PRIVATE ${SYCL_COMMON_COMPILE} ${SYCL_CUDA_BACKEND_OPTS})
target_compile_options(grad_g PRIVATE ${SYCL_COMMON_COMPILE} ${SYCL_CUDA_BACKEND_OPTS})

# --- GANSU ライブラリ ---
set(GANSU_SOURCES
    src_sycl/basis_set.cpp
    src_sycl/parseXYZ.cpp
    src_sycl/hf.dp.cpp
    src_sycl/rhf.dp.cpp
    src_sycl/uhf.dp.cpp
    src_sycl/rohf.dp.cpp
    src_sycl/gpu_manager.dp.cpp
    src_sycl/gpu_kernels.dp.cpp
    src_sycl/profiler.dp.cpp
    src_sycl/parameter_manager.cpp
    src_sycl/eri.dp.cpp
    src_sycl/eri_ri.dp.cpp
    src_sycl/eri_stored.dp.cpp
    src_sycl/eri_stored_fci.dp.cpp
    src_sycl/eri_stored_mp4.dp.cpp
    src_sycl/eri_stored_ump.dp.cpp
    src_sycl/davidson_solver.dp.cpp
    src_sycl/fci.dp.cpp
    src_sycl/test_operators.dp.cpp
    src_sycl/utils.cpp
)
add_library(gansu STATIC ${GANSU_SOURCES})
target_compile_definitions(gansu PRIVATE PROFILE_ENABLED CONSOLE_COLOR_ENABLED)
target_compile_options(gansu PRIVATE ${SYCL_COMMON_COMPILE} ${SYCL_CUDA_BACKEND_OPTS})
target_link_libraries(gansu PUBLIC int1e int2e int2c2e int3c2e int2e_direct grad_w grad_s grad_k grad_v grad_g)
target_link_options(gansu PUBLIC -fsycl -fsycl-targets=nvptx64-nvidia-cuda)

# --- oneMKL GPU libraries ---
link_directories(/home/mikei/docker_env/oneMath/build_9/lib)
target_link_libraries(gansu PUBLIC
    int1e
    int2e
    int2c2e
    int3c2e
    int2e_direct
    onemath
)

# --- oneAPI MKL リンク（Sparse は除く） ---
#find_package(MKL REQUIRED)

# --- 実行ファイル ---
add_executable(HF_main src_sycl/HF_main.dp.cpp)
target_compile_options(HF_main PRIVATE ${SYCL_COMMON_COMPILE} ${SYCL_CUDA_BACKEND_OPTS})
target_compile_definitions(HF_main PRIVATE PROFILE_ENABLED CONSOLE_COLOR_ENABLED)
target_link_options(HF_main PRIVATE -fsycl -fsycl-targets=nvptx64-nvidia-cuda ${SYCL_CUDA_BACKEND_OPTS})
target_link_libraries(HF_main PRIVATE gansu onemath)

# --- TBB が必要なら ---
# find_package(TBB REQUIRED)
# target_link_libraries(gansu PRIVATE TBB::tbb)


# Add test directory (optional, can be enabled/disabled)
option(BUILD_TESTS "Build the test suite" OFF)
if(BUILD_TESTS)
    add_subdirectory(test_sycl)
endif()


# === GANSU-UI (Web UI) ===
set(UI_SRC_DIR "${CMAKE_SOURCE_DIR}/ui")
set(UI_BUILD_DIR "${CMAKE_BINARY_DIR}/gansu-ui")
set(NODE_VERSION "v22.14.0")

# Find npm; if not found, download Node.js to build/_node/
find_program(NPM_EXECUTABLE npm)
if(NOT NPM_EXECUTABLE)
    set(NODE_DIR "${CMAKE_BINARY_DIR}/_node")
    if(NOT EXISTS "${NODE_DIR}/bin/npm")
        message(STATUS "npm not found. Downloading Node.js ${NODE_VERSION}...")
        if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
            set(NODE_ARCH "arm64")
        else()
            set(NODE_ARCH "x64")
        endif()
        set(NODE_TAR "node-${NODE_VERSION}-linux-${NODE_ARCH}.tar.xz")
        file(DOWNLOAD
            "https://nodejs.org/dist/${NODE_VERSION}/${NODE_TAR}"
            "${CMAKE_BINARY_DIR}/${NODE_TAR}"
            SHOW_PROGRESS
        )
        execute_process(
            COMMAND ${CMAKE_COMMAND} -E tar xf "${CMAKE_BINARY_DIR}/${NODE_TAR}"
            WORKING_DIRECTORY "${CMAKE_BINARY_DIR}"
        )
        file(RENAME "${CMAKE_BINARY_DIR}/node-${NODE_VERSION}-linux-${NODE_ARCH}" "${NODE_DIR}")
        file(REMOVE "${CMAKE_BINARY_DIR}/${NODE_TAR}")
    endif()
    set(NPM_EXECUTABLE "${NODE_DIR}/bin/npm")
    set(NODE_PATH_PREFIX "${NODE_DIR}/bin:")
endif()

# Frontend build (only when source changes)
file(GLOB_RECURSE UI_FRONTEND_SOURCES "${UI_SRC_DIR}/frontend/src/*.ts")
add_custom_command(
    OUTPUT "${UI_BUILD_DIR}/frontend/dist/index.html"
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${UI_SRC_DIR}/frontend" "${UI_BUILD_DIR}/_frontend_build"
    COMMAND ${CMAKE_COMMAND} -E env "PATH=${NODE_PATH_PREFIX}$ENV{PATH}"
        "${NPM_EXECUTABLE}" install --prefix "${UI_BUILD_DIR}/_frontend_build"
    COMMAND ${CMAKE_COMMAND} -E env "PATH=${NODE_PATH_PREFIX}$ENV{PATH}"
        "${NPM_EXECUTABLE}" run build --prefix "${UI_BUILD_DIR}/_frontend_build"
        -- --outDir "${UI_BUILD_DIR}/frontend/dist"
    DEPENDS ${UI_FRONTEND_SOURCES} "${UI_SRC_DIR}/frontend/index.html"
    COMMENT "Building GANSU-UI frontend"
)

# Backend copy + frontend build
add_custom_target(gansu-ui ALL
    DEPENDS "${UI_BUILD_DIR}/frontend/dist/index.html"
    COMMAND ${CMAKE_COMMAND} -E make_directory "${UI_BUILD_DIR}/backend"
    COMMAND ${CMAKE_COMMAND} -E copy
        "${UI_SRC_DIR}/backend/main.py"
        "${UI_SRC_DIR}/backend/runner.py"
        "${UI_SRC_DIR}/backend/parser.py"
        "${UI_SRC_DIR}/backend/requirements.txt"
        "${UI_BUILD_DIR}/backend/"
    COMMAND ${CMAKE_COMMAND} -E copy "${UI_SRC_DIR}/run.sh" "${UI_BUILD_DIR}/"
    COMMENT "Setting up GANSU-UI"
)

# Generate .env with correct paths (only if not already present)
if(NOT EXISTS "${UI_BUILD_DIR}/.env")
    file(WRITE "${UI_BUILD_DIR}/.env"
        "GANSU_PATH=${CMAKE_SOURCE_DIR}\n"
        "HF_MAIN_PATH=${CMAKE_BINARY_DIR}/HF_main\n"
    )
endif()

