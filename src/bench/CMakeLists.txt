#
# Benchmarks/Kernels
#

# Grab all cpp files in `src_bench`
file(GLOB BENCH_FILES *.cpp)

# Loop through all .cpp files
foreach(BENCH_SRC ${BENCH_FILES})
  get_filename_component(BENCH_NAME ${BENCH_SRC} NAME_WE)
  MESSAGE(STATUS "MAKING BENCHMARK ${BENCH_NAME}")

  # 
  add_executable(${BENCH_NAME} ${BENCH_SRC})
  target_link_libraries(
    ${BENCH_NAME}
    PRIVATE 
    fricc
    BLAS::BLAS 
    LAPACK::LAPACK 
    OpenMP::OpenMP_CXX
    TBB::tbb
  )
endforeach()