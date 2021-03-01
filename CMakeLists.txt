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
set(CMAKE_CXX_FLAGS_RELEASE "-g -O3")

#
# Some RPATH settings (more later during install)
#
set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

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
  src/rintermediates.cpp
  src/rccsd.cpp
)

# The actual library
add_library(fricc SHARED ${FRICC_SOURCES})

# Use BLAS/LAPACKE as Eigen backend
target_compile_definitions(fricc PUBLIC -DEIGEN_USE_BLAS -DEIGEN_USE_LAPACKE)
target_link_libraries(fricc PRIVATE BLAS::BLAS LAPACK::LAPACK)


# Installation location
install(
  TARGETS fricc
  LIBRARY DESTINATION ${CMAKE_SOURCE_DIR}/lib
)


#
# Pybind11 Wrappers/Module
#

# Set install RPATH for py_rccsd ONLY
set(CMAKE_INSTALL_RPATH ${CMAKE_SOURCE_DIR}/lib)

# Build and link Python wrappers
pybind11_add_module(py_rccsd ${CMAKE_SOURCE_DIR}/src/py_rccsd.cpp)
target_link_libraries(py_rccsd PRIVATE fricc)

install(
  TARGETS py_rccsd
  LIBRARY DESTINATION ${CMAKE_SOURCE_DIR}/fricc
)