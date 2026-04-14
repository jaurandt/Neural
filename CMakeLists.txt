cmake_minimum_required(VERSION 3.14)
project(Neural CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(neural
    src/main.cpp
    src/TrainingData.cpp
    src/Neuron.cpp
    src/Net.cpp
)

target_include_directories(neural PRIVATE src)
