#include <pybind11/pybind11.h>

int add_numbers(int x, int y) {
    return x + y;
}

PYBIND11_MODULE(example, m) {
    m.def("add_numbers", &add_numbers, "A function that adds two numbers");
}