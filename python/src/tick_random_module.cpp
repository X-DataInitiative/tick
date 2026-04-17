#include <pybind11/pybind11.h>

#include "common/tick_pybind11_arrays.h"
#include "tick/random/test_rand.h"

namespace py = pybind11;

PYBIND11_MODULE(crandom, m) {
  tick::pybind::ensure_numpy_imported();

  m.doc() = "tick.random pybind11 bindings";

  py::class_<Rand>(m, "Rand")
      .def(py::init<>())
      .def(py::init<int>(), py::arg("seed"))
      .def("uniform_int", &Rand::uniform_int, py::arg("a"), py::arg("b"))
      .def("uniform_ulong", &Rand::uniform_ulong, py::arg("a"), py::arg("b"))
      .def("uniform", static_cast<double (Rand::*)()>(&Rand::uniform))
      .def("uniform", static_cast<double (Rand::*)(double, double)>(&Rand::uniform),
           py::arg("a"), py::arg("b"))
      .def("gaussian", static_cast<double (Rand::*)()>(&Rand::gaussian))
      .def("gaussian",
           static_cast<double (Rand::*)(double, double)>(&Rand::gaussian),
           py::arg("mean"), py::arg("std"))
      .def("exponential", &Rand::exponential, py::arg("intensity"))
      .def("poisson", &Rand::poisson, py::arg("rate"))
      .def("set_discrete_dist", &Rand::set_discrete_dist, py::arg("probabilities"))
      .def("discrete", static_cast<ulong (Rand::*)()>(&Rand::discrete))
      .def("discrete", static_cast<ulong (Rand::*)(ArrayDouble)>(&Rand::discrete),
           py::arg("probabilities"))
      .def("get_seed", &Rand::get_seed)
      .def("reseed", &Rand::reseed, py::arg("seed"));

  m.def("test_uniform_int", &test_uniform_int, py::arg("a"), py::arg("b"),
        py::arg("size"), py::arg("seed") = -1);
  m.def("test_uniform", static_cast<SArrayDoublePtr (*)(ulong, int)>(&test_uniform),
        py::arg("size"), py::arg("seed") = -1);
  m.def("test_uniform",
        static_cast<SArrayDoublePtr (*)(double, double, ulong, int)>(&test_uniform),
        py::arg("a"), py::arg("b"), py::arg("size"), py::arg("seed") = -1);
  m.def("test_gaussian", static_cast<SArrayDoublePtr (*)(ulong, int)>(&test_gaussian),
        py::arg("size"), py::arg("seed") = -1);
  m.def("test_gaussian",
        static_cast<SArrayDoublePtr (*)(double, double, ulong, int)>(&test_gaussian),
        py::arg("mean"), py::arg("std"), py::arg("size"),
        py::arg("seed") = -1);
  m.def("test_exponential", &test_exponential, py::arg("intensity"),
        py::arg("size"), py::arg("seed") = -1);
  m.def("test_poisson", &test_poisson, py::arg("rate"), py::arg("size"),
        py::arg("seed") = -1);
  m.def("test_discrete", &test_discrete, py::arg("probabilities"),
        py::arg("size"), py::arg("seed") = -1);
  m.def("test_uniform_lagged", &test_uniform_lagged, py::arg("size"),
        py::arg("wait_time") = 0, py::arg("seed") = -1);
  m.def("test_uniform_threaded", &test_uniform_threaded, py::arg("size"),
        py::arg("wait_time") = 0, py::arg("seed") = -1);
}
