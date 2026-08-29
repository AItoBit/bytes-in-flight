cmake_minimum_required(VERSION 3.24)
project(bytes_in_flight LANGUAGES CXX CUDA)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES native)
endif()

add_executable(bif src/main.cu)
set_target_properties(bif PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/bin")
target_compile_options(bif PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-O3 --ptxas-options=-v -lineinfo>)
