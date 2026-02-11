cmake_minimum_required(VERSION 3.15...3.27)
project(vectorforge_cpp)

set(Python_ROOT_DIR "${CMAKE_SOURCE_DIR}/python/.venv")
set(Python_FIND_VIRTUALENV FIRST)
find_package(Python 3.11 REQUIRED COMPONENTS Interpreter Development)

find_package(pybind11 REQUIRED)

set(SOURCES
    cpp/src/hello.cpp
)

pybind11_add_module(vectorforge_cpp ${SOURCES})

target_compile_features(vectorforge_cpp PRIVATE cxx_std_17)
target_compile_options(vectorforge_cpp PRIVATE -O3 -march=native)

target_include_directories(vectorforge_cpp PRIVATE cpp/include)

install(TARGETS vectorforge_cpp LIBRARY DESTINATION python/vectorforge)
