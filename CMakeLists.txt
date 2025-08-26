cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
project(libtorc_caft_trt CXX)

# --- Find LibTorch ---
find_package(Torch REQUIRED)

# --- Find TensorRT ---
# Set TENSORRT_DIR to your TensorRT root directory, e.g., /usr/src/tensorrt
# You can set it as an environment variable or directly in this file.
if(NOT DEFINED TENSORRT_DIR)
    set(TENSORRT_DIR $ENV{TENSORRT_DIR})
    if(NOT DEFINED TENSORRT_DIR)
        message(FATAL_ERROR "TENSORRT_DIR is not set. Please point it to your TensorRT installation.")
    endif()
endif()

message(STATUS "Using TensorRT from: ${TENSORRT_DIR}")

# Find headers
find_path(TENSORRT_INCLUDE_DIR NvInfer.h
    HINTS ${TENSORRT_DIR}/include
)

# Find libraries
find_library(TENSORRT_INFER_LIBRARY nvinfer HINTS ${TENSORRT_DIR}/lib)
find_library(TENSORRT_ONNX_PARSER_LIBRARY nvonnxparser HINTS ${TENSORRT_DIR}/lib)

if(NOT TENSORRT_INCLUDE_DIR OR NOT TENSORRT_INFER_LIBRARY OR NOT TENSORRT_ONNX_PARSER_LIBRARY)
    message(FATAL_ERROR "Failed to find all required TensorRT components.")
endif()

# Find CUDA (needed for cudaMemcpy, etc.)
find_package(CUDA REQUIRED)

# --- Find fmt ---
find_package(fmt REQUIRED)

# --- Project Configuration ---
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Define source files
set(CAFT_LIB_SRCS
    tensorrt_infer.cpp
    cached_fine_tune_model.cpp
)
set(CAFT_APP_SRCS
    main.cpp
)

# Create the main executable
add_executable(caft_app_trt ${CAFT_APP_SRCS} ${CAFT_LIB_SRCS})

# Link all libraries
target_link_libraries(caft_app_trt
    PRIVATE
    ${TORCH_LIBRARIES}
    fmt::fmt
    ${TENSORRT_INFER_LIBRARY}
    ${TENSORRT_ONNX_PARSER_LIBRARY}
    CUDA::cudart # Link CUDA runtime
)

# Include directories
target_include_directories(caft_app_trt
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${TENSORRT_INCLUDE_DIR}
    ${CUDA_INCLUDE_DIRS}
)