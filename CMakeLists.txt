cmake_minimum_required(VERSION 3.20)
project(smort)

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")

add_executable(smort main.cpp)

target_link_libraries(smort m)
