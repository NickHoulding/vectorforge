#include <pybind11/pybind11.h>
#include <string>

std::string helloWorld() {
    return "Hello World!";
}

PYBIND11_MODULE(vectorforge_cpp, m) {
    m.doc() = "pybind11 vectorforge plugin";
    m.def("helloWorld", &helloWorld, "A function that returns the string: 'Hello World!'");
}
