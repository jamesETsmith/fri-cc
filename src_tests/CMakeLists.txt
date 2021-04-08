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
add_test(NAME get_m_largest_tests COMMAND tests)