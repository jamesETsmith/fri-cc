#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>

// The ordering of the includes is important
#include <fricc.hpp>
#include <rccsd.hpp>
#include <rintermediates.hpp>

namespace py = pybind11;

PYBIND11_MODULE(py_ccsd, m)
{
    m.def("update_amps", &RCCSD::update_amps, "CCSD.update_amps implemented in C++.");
    m.def("make_Foo", &make_Foo, "Nothing");
    m.def("make_Fvv", &make_Fvv, "Nothing");
    m.def("make_Fov", &make_Fov, "Nothing");
    m.def("make_Loo", &make_Loo, "Nothing");
    m.def("make_Lvv", &make_Lvv, "Nothing");
    m.def("make_Woooo", &make_Woooo, "Nothing");
    m.def("make_Wvvvv", &make_Wvvvv, "Nothing");
    m.def("make_Wvoov", &make_Wvoov, "Nothing");
    m.def("make_Wvovo", &make_Wvovo, "Nothing");
}