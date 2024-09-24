cmake_minimum_required(VERSION 3.10)

project(CppML VERSION 1.0)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

find_package(Eigen3 3.3 REQUIRED NO_MODULE)

find_package(Matplot++ REQUIRED)

add_executable(CppML src/main.cpp)

target_sources(CppML PUBLIC src/CSVETL.cpp src/KNN.cpp src/LinearRegression.cpp)

target_link_libraries(CppML Eigen3::Eigen)

target_link_libraries(CppML Matplot++::matplot)

target_include_directories(CppML PUBLIC "${PROJECT_BINARY_DIR}" "${PROJECT_SOURCE_DIR}/include")
