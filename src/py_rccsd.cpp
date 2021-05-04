#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <tuple>

// The ordering of the includes is important
//
// #include <fricc.hpp>
#include <fri_utils.hpp>
#include <rccsd.hpp>
#include <rintermediates.hpp>
#include <sparse_tensor.hpp>

namespace py = pybind11;

PYBIND11_MODULE(py_rccsd, m) {
  m.def("update_amps", &RCCSD::update_amps,
        "CCSD.update_amps implemented in C++.");
  m.def("make_Foo", &make_Foo, "Nothing");
  m.def("make_Fvv", &make_Fvv, "Nothing");
  m.def("make_Fov", &make_Fov, "Nothing");
  m.def("make_Loo", &make_Loo, "Nothing");
  m.def("make_Lvv", &make_Lvv, "Nothing");
  m.def("make_Woooo", &make_Woooo, "Nothing");
  m.def("make_Wvvvv", &make_Wvvvv, "Nothing");
  m.def("make_Wvoov", &make_Wvoov, "Nothing");
  m.def("make_Wvovo", &make_Wvovo, "Nothing");
  //
  m.def("get_m_largest", &get_m_largest,
        "Get the m largest elements of a vector.");
  m.def("parallel_sort", &my_parallel_sort, "");

  py::class_<SparseTensor4d>(m, "SparseTensor4d")
      .def(py::init<std::array<size_t, 4>, double>())
      .def(py::init<Eigen::Ref<Eigen::VectorXd>, std::array<size_t, 4>,
                    const size_t>())
      .def("get_element",
           [](SparseTensor4d& sp_tensor, const size_t mi) {
             std::array<size_t, 4> idx_arr;
             double value;
             sp_tensor.get_element(mi, idx_arr, value);
             return std::make_tuple(idx_arr, value);
           })
      .def("print", &SparseTensor4d::print);

  // Contraction wrapper
  m.def("contract_DTSpT", &contract_SparseTensor4d_wrapper);
  //   m.def("init_sparse_tensor", &init_sparse_tensor, "Nothing");
}