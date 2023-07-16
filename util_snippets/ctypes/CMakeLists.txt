cmake_minimum_required(VERSION 3.10)

project(example C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

add_library(example SHARED example.c)

set_target_properties(example PROPERTIES OUTPUT_NAME "example")

install(TARGETS example DESTINATION lib)
install(FILES example.h DESTINATION include)