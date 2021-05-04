cmake_minimum_required(VERSION 3.15)
project(fricc LANGUAGES CXX)



#
# Compiler standards and options
#
set(CMAKE_CXX_STANDARD 17) 
set(CMAKE_CXX_STANDARD_REQUIRED ON)

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

set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=${ARCH} -ffast-math -Wno-narrowing")
set(CMAKE_CXX_FLAGS_DEBUG "-g")


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
find_package(TBB REQUIRED)
message(STATUS "Intel TBB Found? ${TBB_tbb_FOUND}")

# Eigen
include_directories(${CMAKE_SOURCE_DIR}/external/eigen)

# Pybind11
add_subdirectory(${CMAKE_SOURCE_DIR}/external/pybind11)

#
# Libfricc
#
include_directories(${CMAKE_SOURCE_DIR}/include)

set(FRICC_SOURCES 
  src/fri_utils.cpp
  src/rintermediates.cpp
  src/rccsd.cpp
  src/sparse_tensor.cpp
)

# The actual library
add_library(fricc SHARED ${FRICC_SOURCES})


# Use BLAS/LAPACKE as Eigen backend
target_compile_definitions(fricc PRIVATE -DEIGEN_USE_BLAS -DEIGEN_USE_LAPACKE)
target_link_libraries(fricc PRIVATE BLAS::BLAS LAPACK::LAPACK OpenMP::OpenMP_CXX TBB::tbb)


# Installation location
install(TARGETS fricc LIBRARY DESTINATION ${CMAKE_SOURCE_DIR}/lib)

#
# C++ Tests
#
if (${CPP_TEST} STREQUAL ON)
  message(STATUS "Building C++ tests. After building, run from the command line with:\n\tmake test")
  enable_testing()
  add_subdirectory(src_tests)

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
target_link_libraries(py_rccsd  PRIVATE   fricc)

install(TARGETS py_rccsd LIBRARY DESTINATION ${CMAKE_SOURCE_DIR}/fricc)