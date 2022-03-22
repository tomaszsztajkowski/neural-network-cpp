cmake_minimum_required(VERSION 3.21)
project(network)

set(CMAKE_CXX_STANDARD 14)

add_executable(network main.cpp neural.cpp matrix.cpp)
