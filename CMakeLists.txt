cmake_minimum_required(VERSION 3.15)
project(fricc LANGUAGES CXX)



#
# Compiler standards and options
#
set(CMAKE_CXX_STANDARD 20 CACHE STRING "C++ version selection")  # or 11, 14, 17, 20
set(CMAKE_CXX_STANDARD_REQUIRED ON)  # optional, ensure standard is supported

set(CMAKE_BUILD_TYPE "RELEASE" CACHE STRING "Build type")
set(CPP_TEST ON CACHE BOOL "Build C++ Tests")

# Set difference between DEBUG and RELEASE
set(CMAKE_CXX_FLAGS_DEBUG "-g")

if (CMAKE_BUILD_TYPE STREQUAL "RELEASE")
execute_process(
  COMMAND cat /sys/devices/cpu/caps/pmu_name
  OUTPUT_VARIABLE ARCH
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
message(STATUS "Using the compiler optimizations for arch=${ARCH}")
endif()

set(CMAKE_CXX_FLAGS_RELEASE "-g -O3 -march=${ARCH} -ffast-math")

#
# Some RPATH settings (more later during install)
#
set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

#
# Dependencies
#

# BLAS and LAPACK
find_package(OpenMP REQUIRED)
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
  src/rintermediates.cpp
  src/rccsd.cpp
  src/fri_utils.cpp
)

# The actual library
add_library(fricc SHARED ${FRICC_SOURCES})

# Use BLAS/LAPACKE as Eigen backend
target_compile_definitions(fricc PUBLIC -DEIGEN_USE_BLAS -DEIGEN_USE_LAPACKE)
target_link_libraries(
  fricc 
  PRIVATE 
  BLAS::BLAS 
  LAPACK::LAPACK 
  OpenMP::OpenMP_CXX
  tbb
  )


# Installation location
install(
  TARGETS fricc
  LIBRARY DESTINATION ${CMAKE_SOURCE_DIR}/lib
)

#
# C++ Tests
#
if (${CPP_TEST} STREQUAL "ON")
  message(STATUS "Building C++ tests. After building, run from the command line with:\n\t./tests")
  enable_testing()
  include(${CMAKE_SOURCE_DIR}/external/doctest/scripts/cmake/doctest.cmake)

  # Make test executable
  add_executable(tests ${CMAKE_SOURCE_DIR}/src_tests/main.cpp)
  target_include_directories(tests PUBLIC ${CMAKE_SOURCE_DIR}/external/doctest/doctest)
  target_link_libraries(
    tests 
    PRIVATE 
    fricc
    BLAS::BLAS 
    LAPACK::LAPACK 
    OpenMP::OpenMP_CXX
    tbb
  )

else()
message(STATUS "Skipping build of C++ tests. To set it run:\n\tcmake .. -DCPP_TEST=ON")
endif()

#
# Pybind11 Wrappers/Module
#

# Set install RPATH for py_rccsd ONLY
set(CMAKE_INSTALL_RPATH ${CMAKE_SOURCE_DIR}/lib)

# Build and link Python wrappers
pybind11_add_module(py_rccsd ${CMAKE_SOURCE_DIR}/src/py_rccsd.cpp)
target_link_libraries(py_rccsd  PRIVATE   fricc   )

install(
  TARGETS py_rccsd
  LIBRARY DESTINATION ${CMAKE_SOURCE_DIR}/fricc
)