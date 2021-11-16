cmake_minimum_required(VERSION 3.15)
project(fricc LANGUAGES CXX)



#
# Compiler standards and options
#
set(CMAKE_CXX_STANDARD 20) 
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON) # For clang-tidy


# Build type defaults to release
if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release")
endif()

# Build CPP tests by default
if (NOT CPP_TEST)
  set(CPP_TEST OFF)
endif()

# Special compile flags (i.e. optimizations) for release builds
if (CMAKE_BUILD_TYPE STREQUAL "Release")
  execute_process(
    COMMAND cat /sys/devices/cpu/caps/pmu_name
    OUTPUT_VARIABLE ARCH
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )
  if (ARCH STREQUAL "")
    message(WARNING "Architecture not found, using arch=native.")
    set(ARCH "native")
  else()
    message(STATUS "Using the compiler optimizations for arch=${ARCH}")
  endif()
endif()

set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS} -O3 -g -march=${ARCH} -Wno-narrowing")
message(STATUS "Using the following C++ compiler flags\n\t${CMAKE_CXX_FLAGS_RELEASE}")

# User libc++ with Clang
if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libc++")
endif()

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
# set(TBB_DIR ${CMAKE_SOURCE_DIR}/external/oneTBB/install/lib64/cmake/TBB/)
# find_package(TBB)
# message(STATUS "Intel TBB Found? ${TBB_tbb_FOUND}")

# Eigen
include_directories(${CMAKE_SOURCE_DIR}/external/eigen)

# Sorting Sandbox
include_directories(${CMAKE_SOURCE_DIR}/external/sorting_sandbox/include)

# Pybind11
add_subdirectory(${CMAKE_SOURCE_DIR}/external/pybind11)
# include_directories(${CMAKE_SOURCE_DIR}/external/pybind11/include)

#
# Libfricc
#
include_directories(${CMAKE_SOURCE_DIR}/include)

set(FRICC_SOURCES 
  src/fri_utils.cpp
)

# The actual library
add_library(fricc SHARED ${FRICC_SOURCES})


# Use BLAS/LAPACKE as Eigen backend
target_compile_definitions(fricc PRIVATE -DEIGEN_USE_BLAS -DEIGEN_USE_LAPACKE)
target_link_libraries(fricc PRIVATE BLAS::BLAS LAPACK::LAPACK OpenMP::OpenMP_CXX)


# Installation location
install(TARGETS fricc LIBRARY DESTINATION ${CMAKE_SOURCE_DIR}/lib)

#
# C++ Tests
#
if (${CPP_TEST} STREQUAL ON)
  message(STATUS "Building C++ tests. After building, run from the command line with:\n\tmake test")
  enable_testing()
  add_subdirectory(src/tests)

else()
  message(STATUS "Skipping build of C++ tests. To set it run:\n\tcmake .. -DCPP_TEST=ON")
endif()


#
# Pybind11 Wrappers/Module
#

# Set install RPATH for py_rccsd ONLY
set(CMAKE_INSTALL_RPATH ${CMAKE_SOURCE_DIR}/lib)

# Build and link Python wrappers
pybind11_add_module(py_rccsd ${CMAKE_SOURCE_DIR}/src/py_rccsd.cpp src/sparse_tensor.cpp)
target_link_libraries(py_rccsd  PRIVATE fricc OpenMP::OpenMP_CXX)

install(TARGETS py_rccsd LIBRARY DESTINATION ${CMAKE_SOURCE_DIR}/fricc)