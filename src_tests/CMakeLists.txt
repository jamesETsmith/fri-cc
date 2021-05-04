#
# Collect Tests and Make Test Executables
#

file(GLOB TEST_FILES test_*.cpp)

foreach(TEST_SRC ${TEST_FILES})


  get_filename_component(TEST_NAME ${TEST_SRC} NAME_WE)
  MESSAGE(STATUS "MAKING TEST ${TEST_NAME}")

  add_executable(${TEST_NAME} ${TEST_SRC})
  target_include_directories(
    ${TEST_NAME} PUBLIC 
    ${CMAKE_SOURCE_DIR}/external/doctest/doctest
    )
  target_link_libraries(
    ${TEST_NAME}
    PRIVATE 
    fricc
    BLAS::BLAS 
    LAPACK::LAPACK 
    OpenMP::OpenMP_CXX
    TBB::tbb
  )
  add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME})

endforeach()



