cmake_minimum_required(VERSION 3.19...3.30)
project(intextus_backend LANGUAGES C CXX)

if(POLICY CMP0169)
    cmake_policy(SET CMP0169 OLD)
endif()

# FORCE PIC FOR ALL DEPENDENCIES (Fixes the R_X86_64_PC32 linker error)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Use standard C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)

# 1. Fetch tokenizers-cpp
FetchContent_Declare(
    tokenizers-cpp
    GIT_REPOSITORY https://github.com/mlc-ai/tokenizers-cpp.git
    GIT_TAG        main
    GIT_SHALLOW    TRUE
)

FetchContent_GetProperties(tokenizers-cpp)
if(NOT tokenizers-cpp_POPULATED)
    FetchContent_Populate(tokenizers-cpp)
    
    # Patch tokenizers-cpp/CMakeLists.txt to not hardcode aarch64-unknown-linux-gnu cross-compilers
    # since we are compiling inside native aarch64 docker containers under QEMU emulation.
    set(TC_CMAKE_FILE "${tokenizers-cpp_SOURCE_DIR}/CMakeLists.txt")
    if(EXISTS "${TC_CMAKE_FILE}")
        file(READ "${TC_CMAKE_FILE}" TC_CMAKE_CONTENT)
        string(REPLACE
            "AR_\${TOKENIZERS_CPP_CARGO_TARGET}=\${TOOLCHAIN_DIR}\${TOKENIZERS_CPP_CARGO_TARGET}-ar"
            "AR_\${TOKENIZERS_CPP_CARGO_TARGET}=ar"
            TC_CMAKE_CONTENT "${TC_CMAKE_CONTENT}")
        string(REPLACE
            "CC_\${TOKENIZERS_CPP_CARGO_TARGET}=\${TOOLCHAIN_DIR}\${TOKENIZERS_CPP_CARGO_TARGET}-gcc"
            "CC_\${TOKENIZERS_CPP_CARGO_TARGET}=gcc"
            TC_CMAKE_CONTENT "${TC_CMAKE_CONTENT}")
        string(REPLACE
            "CXX_\${TOKENIZERS_CPP_CARGO_TARGET}=\${TOOLCHAIN_DIR}\${TOKENIZERS_CPP_CARGO_TARGET}-g++"
            "CXX_\${TOKENIZERS_CPP_CARGO_TARGET}=g++"
            TC_CMAKE_CONTENT "${TC_CMAKE_CONTENT}")
        file(WRITE "${TC_CMAKE_FILE}" "${TC_CMAKE_CONTENT}")
    endif()

    add_subdirectory(${tokenizers-cpp_SOURCE_DIR} ${tokenizers-cpp_BINARY_DIR})
endif()

# 2. Fetch llama.cpp
# Disable examples/tests to speed up build
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "Disable tests" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "Disable examples" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "Disable server" FORCE)
set(GGML_OPENMP OFF CACHE BOOL "Disable OpenMP" FORCE)
set(GGML_NATIVE ON CACHE BOOL "Enable native optimization")
set(GGML_METAL OFF CACHE BOOL "Disable Metal" FORCE)
set(LLAMA_METAL OFF CACHE BOOL "Disable Metal" FORCE)

FetchContent_Declare(
    llama_cpp
    GIT_REPOSITORY https://github.com/ggml-org/llama.cpp.git
    GIT_TAG        b8500 # Stable release tag
    GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(llama_cpp)

# 3. Locate nanobind automatically using your current Python environment
find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
execute_process(
    COMMAND "${Python_EXECUTABLE}" -c "import nanobind; print(nanobind.cmake_dir())"
    OUTPUT_VARIABLE nanobind_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
find_package(nanobind REQUIRED)

# 4. Define our C++ extension module
nanobind_add_module(
    _core                         # The name of the module
    src/intextus/_core.cpp        # Where the C++ source code is
)

# 5. Link libraries
target_link_libraries(_core PRIVATE tokenizers_cpp llama)

# Configure RPATH for the extension module to find libraries in the same directory
if(APPLE)
    set_target_properties(_core PROPERTIES
        INSTALL_RPATH "@loader_path;@loader_path/../lib"
        BUILD_WITH_INSTALL_RPATH TRUE
    )
elseif(UNIX)
    set_target_properties(_core PROPERTIES
        INSTALL_RPATH "$ORIGIN;$ORIGIN/../lib"
        BUILD_WITH_INSTALL_RPATH TRUE
    )
endif()

# 6. Tell the build system to drop the finished binary inside your python package folder
install(TARGETS _core DESTINATION intextus)