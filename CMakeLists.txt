cmake_minimum_required(VERSION 3.5)

project(cudaSift)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 11)

# Basic version info
set(cudaSift_VERSION_MAJOR 2)
set(cudaSift_VERSION_MINOR 0)
set(cudaSift_VERSION_PATCH 0)

# Let CMake find packages normally without too much manual configuration
find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)

find_program(NVIDIA_SMI "nvidia-smi")
set(CUDA_ARCH_FLAG "-arch=sm_61")  # 默认值

if(NVIDIA_SMI)
    execute_process(
        COMMAND ${NVIDIA_SMI} --query-gpu=compute_cap --format=csv,noheader,nounits
        OUTPUT_VARIABLE GPU_COMPUTE_CAP
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
        RESULT_VARIABLE SMI_RESULT
    )
    
    if(SMI_RESULT EQUAL 0 AND GPU_COMPUTE_CAP)
        # 取第一行（第一个GPU的架构）
        string(REGEX MATCH "^[0-9]+\\.[0-9]+" FIRST_GPU_CAP "${GPU_COMPUTE_CAP}")
        if(FIRST_GPU_CAP)
            string(REPLACE "." "" ARCH_NUM "${FIRST_GPU_CAP}")
            set(CUDA_ARCH_FLAG "-arch=sm_${ARCH_NUM}")
            message(STATUS "Auto-detected CUDA architecture: sm_${ARCH_NUM}")
        endif()
    endif()
endif()

if(CUDA_ARCH_FLAG STREQUAL "-arch=sm_61")
    message(STATUS "Using default CUDA architecture: sm_61")
endif()

# Find pybind11 with automatic path detection
execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import pybind11; print(pybind11.get_cmake_dir())"
    OUTPUT_VARIABLE PYBIND11_CMAKE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
    RESULT_VARIABLE PYBIND11_RESULT
)

if(PYBIND11_RESULT EQUAL 0 AND EXISTS "${PYBIND11_CMAKE_DIR}")
    message(STATUS "Found pybind11 cmake dir: ${PYBIND11_CMAKE_DIR}")
    set(pybind11_DIR "${PYBIND11_CMAKE_DIR}")
    find_package(pybind11 REQUIRED)
else()
    message(STATUS "Could not auto-detect pybind11 cmake dir, trying fallback methods")
    # Try common conda locations
    set(PYBIND11_SEARCH_PATHS
        "$ENV{CONDA_PREFIX}/lib/python${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR}/site-packages/pybind11/share/cmake/pybind11"
        "$ENV{CONDA_PREFIX}/share/cmake/pybind11"
        "/usr/local/share/cmake/pybind11"
    )
    
    foreach(SEARCH_PATH ${PYBIND11_SEARCH_PATHS})
        if(EXISTS "${SEARCH_PATH}")
            message(STATUS "Found pybind11 at: ${SEARCH_PATH}")
            set(pybind11_DIR "${SEARCH_PATH}")
            break()
        endif()
    endforeach()
    
    find_package(pybind11 REQUIRED)
endif()

if (NOT CUDA_FOUND)
  message(STATUS "CUDA not found. Project will not be built.")
endif(NOT CUDA_FOUND)

# Platform-specific settings
if (WIN32)
  set(EXTRA_CXX_FLAGS "/DVERBOSE /D_CRT_SECURE_NO_WARNINGS ")
  list(APPEND CUDA_NVCC_FLAGS "${CUDA_NVCC_ARCH_FLAGS};--compiler-options;-O2;-DVERBOSE") 
endif()
if (UNIX)
  if (APPLE)
    set(EXTRA_CXX_FLAGS "-DVERBOSE -msse2")
    list(APPEND CUDA_NVCC_FLAGS "${CUDA_NVCC_ARCH_FLAGS};--compiler-options;-O2;-DVERBOSE") 
  else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -msse2 ")
    list(APPEND CUDA_NVCC_FLAGS "-lineinfo;-ccbin;/usr/bin/g++;--compiler-options;-O2;-D_FORCE_INLINES;-DVERBOSE_NOT") 
  endif()
endif()

# Source files
set(cuda_sources
  cudaImage.cu  
  cudaImage.h  
  cudaSiftH.cu 
  cudaSiftH.h  
  matching.cu  
  cudaSiftD.h  
  cudaSift.h  
  cudautils.h
)  

set(sources
  geomFuncs.cpp  
  mainSift.cpp  
)

set(python_binding_sources
  python_binding.cpp
)

include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR} 
  ${OpenCV_INCLUDE_DIRS}
)

# Original executable (optional)
cuda_add_executable(cudasift ${cuda_sources} ${sources} OPTIONS ${CUDA_NVCC_ARCH_FLAGS})

set_target_properties(cudasift PROPERTIES
  COMPILE_FLAGS "${EXTRA_CXX_FLAGS}"			   
)

target_link_libraries(cudasift ${CUDA_cudadevrt_LIBRARY} ${OpenCV_LIBS})

# Create CUDA library for Python binding
# Use the same NVCC flags as the main executable but ensure PIC is enabled
set(CUDA_LIB_NVCC_FLAGS ${CUDA_NVCC_FLAGS})
if(UNIX AND NOT APPLE)
    # Ensure -fPIC is in the compiler options for the library
    list(APPEND CUDA_LIB_NVCC_FLAGS "--compiler-options;-fPIC")
endif()

cuda_add_library(cudasift_lib STATIC ${cuda_sources} geomFuncs.cpp OPTIONS ${CUDA_LIB_NVCC_FLAGS})

set_target_properties(cudasift_lib PROPERTIES
  COMPILE_FLAGS "${EXTRA_CXX_FLAGS}"
  POSITION_INDEPENDENT_CODE ON
)

# Ensure all C++ files are compiled with -fPIC
if(UNIX)
    target_compile_options(cudasift_lib PRIVATE -fPIC)
endif()

target_link_libraries(cudasift_lib ${OpenCV_LIBS})

# Create Python extension module
pybind11_add_module(cuda_sift ${python_binding_sources})

# Set properties for Python module
set_target_properties(cuda_sift PROPERTIES
  COMPILE_FLAGS "${EXTRA_CXX_FLAGS}"
)

# Include directories for Python module
target_include_directories(cuda_sift PRIVATE 
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${OpenCV_INCLUDE_DIRS}
  ${CUDA_INCLUDE_DIRS}
)

# Link libraries for Python module
target_link_libraries(cuda_sift PRIVATE 
  cudasift_lib
  ${OpenCV_LIBS}
  ${CUDA_LIBRARIES}
  ${CUDA_cudadevrt_LIBRARY}
)

# Platform-specific compile options
if(UNIX)
  target_compile_options(cuda_sift PRIVATE -fPIC)
endif()

# Debug output
message(STATUS "Python3_EXECUTABLE: ${Python3_EXECUTABLE}")
message(STATUS "Python3_INCLUDE_DIRS: ${Python3_INCLUDE_DIRS}")
message(STATUS "Python3_LIBRARIES: ${Python3_LIBRARIES}")
message(STATUS "Python3_VERSION: ${Python3_VERSION}")
message(STATUS "pybind11_INCLUDE_DIRS: ${pybind11_INCLUDE_DIRS}")

# Installation
install(FILES 
  ${cuda_sources} 
  ${sources}
  ${python_binding_sources}
  cudaSiftD.cu
  CMakeLists.txt
  Copyright.txt
  DESTINATION .
)