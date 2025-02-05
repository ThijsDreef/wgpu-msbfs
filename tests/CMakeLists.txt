FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG b514bdc898e2951020cbdca1304b75f5950d1f59
)

FetchContent_MakeAvailable(googletest)

add_executable(test_wgpumsbfs test.cpp)

target_link_libraries(test_wgpumsbfs
  gtest
  ${LIB_NAME}
)

add_test(
  NAME tests
  COMMAND test_wgpumsbfs
)
