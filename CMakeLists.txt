cmake_minimum_required(VERSION 3.15)
project(fricc LANGUAGES CXX)

#
# Compiler standards and options
#
set(CMAKE_CXX_STANDARD 20 CACHE STRING "C++ version selection")  # or 11, 14, 17, 20
set(CMAKE_CXX_STANDARD_REQUIRED ON)  # optional, ensure standard is supported

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Debug)
endif()

# Set difference between DEBUG and RELEASE
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

#
# Dependencies
#

# BLAS and LAPACK
find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)

# Eigen
include_directories(${CMAKE_SOURCE_DIR}/external/eigen)

# Pybind11
add_subdirectory(${CMAKE_SOURCE_DIR}/external/pybind11)

#
# Libfricc
#
include_directories(${CMAKE_SOURCE_DIR}/include)

set(FRICC_SOURCES 
    src/ccsd.cpp
    src/rintermediates.cpp
)

# The actual library
add_library(fricc SHARED ${FRICC_SOURCES})

# Use BLAS/LAPACKE as Eigen backend
target_compile_definitions(fricc PUBLIC -DEIGEN_USE_BLAS -DEIGEN_USE_LAPACKE)
target_link_libraries(fricc PRIVATE BLAS::BLAS LAPACK::LAPACK)

#
# Pybind11 Wrappers/Module
#
pybind11_add_module(py_ccsd ${CMAKE_SOURCE_DIR}/src/py_ccsd.cpp)
target_link_libraries(py_ccsd PRIVATE fricc)
