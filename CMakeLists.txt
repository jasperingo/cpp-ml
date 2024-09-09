cmake_minimum_required(VERSION 3.10)

project(CppML VERSION 1.0)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

find_package(Eigen3 3.3 REQUIRED NO_MODULE)

add_executable(CppML src/main.cpp)

target_sources(CppML PUBLIC src/KNN.cpp src/CSVETL.cpp)

target_link_libraries(CppML Eigen3::Eigen)

target_include_directories(CppML PUBLIC "${PROJECT_BINARY_DIR}" "${PROJECT_SOURCE_DIR}/include")
