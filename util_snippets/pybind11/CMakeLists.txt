cmake_minimum_required(VERSION 3.0)


# dynamicaly find pybind11;s path
execute_process(
    COMMAND python -m pybind11 --cmakedir
    OUTPUT_VARIABLE PYBIND11_PATH
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
message("Pybind11 Found at: ${PYBIND11_PATH}")
set(pybind11_DIR ${PYBIND11_PATH})

project(pybind_example)
find_package(pybind11 REQUIRED)
pybind11_add_module(example example.cpp)


# mkdir build && cd build && cmake .. && make