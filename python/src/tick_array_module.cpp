#include <pybind11/pybind11.h>

#include "common/tick_pybind11_arrays.h"
#include "tick/array/serializer.h"

namespace py = pybind11;

PYBIND11_MODULE(array, m) {
  tick::pybind::ensure_numpy_imported();

  m.doc() = "tick.array pybind11 serialization bindings";

  m.def("tick_float_array_to_file", &tick_float_array_to_file, py::arg("_file"),
        py::arg("array"));
  m.def("tick_float_array2d_to_file", &tick_float_array2d_to_file,
        py::arg("_file"), py::arg("array"));
  m.def("tick_float_sparse2d_to_file", &tick_float_sparse2d_to_file,
        py::arg("_file"), py::arg("array"));
  m.def("tick_double_array_to_file", &tick_double_array_to_file,
        py::arg("_file"), py::arg("array"));
  m.def("tick_double_array2d_to_file", &tick_double_array2d_to_file,
        py::arg("_file"), py::arg("array"));
  m.def("tick_double_sparse2d_to_file", &tick_double_sparse2d_to_file,
        py::arg("_file"), py::arg("array"));

  m.def("tick_float_array_from_file", &tick_float_array_from_file,
        py::arg("_file"));
  m.def("tick_float_array2d_from_file", &tick_float_array2d_from_file,
        py::arg("_file"));
  m.def("tick_float_sparse2d_from_file", &tick_float_sparse2d_from_file,
        py::arg("_file"));
  m.def("tick_double_array_from_file", &tick_double_array_from_file,
        py::arg("_file"));
  m.def("tick_double_array2d_from_file", &tick_double_array2d_from_file,
        py::arg("_file"));
  m.def("tick_double_sparse2d_from_file", &tick_double_sparse2d_from_file,
        py::arg("_file"));

  m.def("tick_float_colmaj_sparse2d_to_file", &tick_float_colmaj_sparse2d_to_file,
        py::arg("_file"), py::arg("array"));
  m.def("tick_float_colmaj_sparse2d_from_file",
        &tick_float_colmaj_sparse2d_from_file, py::arg("_file"));
  m.def("tick_double_colmaj_sparse2d_to_file",
        &tick_double_colmaj_sparse2d_to_file, py::arg("_file"),
        py::arg("array"));
  m.def("tick_double_colmaj_sparse2d_from_file",
        &tick_double_colmaj_sparse2d_from_file, py::arg("_file"));
  m.def("tick_float_colmaj_array2d_to_file", &tick_float_colmaj_array2d_to_file,
        py::arg("_file"), py::arg("array"));
  m.def("tick_float_colmaj_array2d_from_file",
        &tick_float_colmaj_array2d_from_file, py::arg("_file"));
  m.def("tick_double_colmaj_array2d_to_file",
        &tick_double_colmaj_array2d_to_file, py::arg("_file"),
        py::arg("array"));
  m.def("tick_double_colmaj_array2d_from_file",
        &tick_double_colmaj_array2d_from_file, py::arg("_file"));
}
