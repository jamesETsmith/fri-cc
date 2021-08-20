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
using namespace pybind11::literals;

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
  // m.def("get_m_largest", &get_m_largest,
  //       "Get the m largest elements of a vector.");
  // m.def("parallel_sort", &my_parallel_sort, "");
  // m.def("parallel_partial_sort", &my_partial_sort, "");
  m.def("partial_argsort", &partial_argsort, "");
  m.def("partial_argsort_paired", &partial_argsort_paired, "");
  m.def("sample_pivotal", &sample_pivotal, "", "n_sample"_a,
        "probs"_a.noconvert());
  m.def("sample_systematic", &sample_systematic, "", "n_sample"_a,
        "probs"_a.noconvert());
  m.def("parallel_sample", &parallel_sample, "", "n_sample"_a,
        "probs"_a.noconvert());

  m.def("make_probability_vector", &make_probability_vector, "",
        "x"_a.noconvert(), "n_sample"_a, "D"_a, "remaining_norm"_a.noconvert());

  m.def("get_d_largest", &get_d_largest, "", "n_sample"_a, "x"_a.noconvert());
  m.def("fri_compression", &fri_compression, "", "x"_a.noconvert(),
        "n_sample"_a, "sampling_method"_a = "pivotal", "verbose"_a = false);

  py::class_<SparseTensor4d>(m, "SparseTensor4d")
      .def(py::init<std::array<size_t, 4>, double>())
      .def(py::init<const std::vector<double>&, std::array<size_t, 4>,
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