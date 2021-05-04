cmake_minimum_required(VERSION 3.15)
project(parallel_sort LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20) 
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "-O3")


find_package(OpenMP REQUIRED)
find_package(TBB REQUIRED)

include_directories(${CMAKE_SOURCE_DIR}/../external/eigen)

add_executable(parallel_sort parallel_sort.cpp)
target_link_libraries(parallel_sort PRIVATE OpenMP::OpenMP_CXX TBB::tbb)
