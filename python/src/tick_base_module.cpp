#include <stdexcept>
#include <string>

#include <pybind11/pybind11.h>

#include "common/tick_pybind11_arrays.h"
#include "tick/base/base_test.h"
#include "tick/base/exceptions_test.h"
#include "tick/base/math/normal_distribution.h"
#include "tick/base/serialization.h"
#include "tick/base/time_func.h"

namespace py = pybind11;

namespace {

void bind_time_function(py::module_ &m) {
  auto cls = py::class_<TimeFunction>(m, "TimeFunction");

  auto inter_mode = py::enum_<TimeFunction::InterMode>(cls, "InterMode");
  inter_mode.value("InterLinear", TimeFunction::InterMode::InterLinear)
      .value("InterConstRight", TimeFunction::InterMode::InterConstRight)
      .value("InterConstLeft", TimeFunction::InterMode::InterConstLeft);

  auto border_type = py::enum_<TimeFunction::BorderType>(cls, "BorderType");
  border_type.value("Border0", TimeFunction::BorderType::Border0)
      .value("BorderConstant", TimeFunction::BorderType::BorderConstant)
      .value("BorderContinue", TimeFunction::BorderType::BorderContinue)
      .value("Cyclic", TimeFunction::BorderType::Cyclic);

  cls.def(py::init<double>(), py::arg("y") = 0.0)
      .def(py::init<const ArrayDouble &, TimeFunction::BorderType,
                    TimeFunction::InterMode, double, double>(),
           py::arg("Y"), py::arg("type"), py::arg("mode"),
           py::arg("dt"), py::arg("border_value"))
      .def(py::init<const ArrayDouble &, const ArrayDouble &, double>(),
           py::arg("T"), py::arg("Y"), py::arg("dt"))
      .def(py::init<const ArrayDouble &, const ArrayDouble &,
                    TimeFunction::BorderType, TimeFunction::InterMode,
                    double, double>(),
           py::arg("T"), py::arg("Y"),
           py::arg("type") = TimeFunction::DEFAULT_BORDER,
           py::arg("mode") = TimeFunction::DEFAULT_INTER, py::arg("dt") = 0.0,
           py::arg("border_value") = 0.0)
      .def("get_inter_mode", &TimeFunction::get_inter_mode)
      .def("get_border_type", &TimeFunction::get_border_type)
      .def("get_sampled_y", &TimeFunction::get_sampled_y)
      .def("get_sample_size", &TimeFunction::get_sample_size)
      .def("get_future_max", &TimeFunction::get_future_max)
      .def("get_t0", &TimeFunction::get_t0)
      .def("set_t0", &TimeFunction::set_t0)
      .def("get_dt", &TimeFunction::get_dt)
      .def("get_support_right", &TimeFunction::get_support_right)
      .def("value",
           static_cast<double (TimeFunction::*)(double)>(&TimeFunction::value),
           py::arg("t"))
      .def("value",
           static_cast<SArrayDoublePtr (TimeFunction::*)(ArrayDouble &)>(
               &TimeFunction::value),
           py::arg("t"))
      .def("primitive", &TimeFunction::primitive, py::arg("t"))
      .def("future_bound",
           static_cast<double (TimeFunction::*)(double)>(
               &TimeFunction::future_bound),
           py::arg("t"))
      .def("future_bound",
           static_cast<SArrayDoublePtr (TimeFunction::*)(ArrayDouble &)>(
               &TimeFunction::future_bound),
           py::arg("t"))
      .def("compute_future_max", &TimeFunction::compute_future_max)
      .def("max_error", &TimeFunction::max_error, py::arg("t"))
      .def("get_norm", &TimeFunction::get_norm)
      .def("get_border_value", &TimeFunction::get_border_value)
      .def(
          py::pickle(
              [](const TimeFunction &time_function) {
                return py::make_tuple(tick::object_to_string(
                    const_cast<TimeFunction *>(&time_function)));
              },
              [](py::tuple state) {
                if (state.size() != 1) {
                  throw std::runtime_error("Invalid TimeFunction pickle state");
                }
                TimeFunction time_function(0.0);
                tick::object_from_string(&time_function,
                                         state[0].cast<std::string>());
                return time_function;
              }));

  cls.attr("InterMode_InterLinear") =
      py::cast(TimeFunction::InterMode::InterLinear);
  cls.attr("InterMode_InterConstRight") =
      py::cast(TimeFunction::InterMode::InterConstRight);
  cls.attr("InterMode_InterConstLeft") =
      py::cast(TimeFunction::InterMode::InterConstLeft);
  cls.attr("BorderType_Border0") =
      py::cast(TimeFunction::BorderType::Border0);
  cls.attr("BorderType_BorderConstant") =
      py::cast(TimeFunction::BorderType::BorderConstant);
  cls.attr("BorderType_BorderContinue") =
      py::cast(TimeFunction::BorderType::BorderContinue);
  cls.attr("BorderType_Cyclic") =
      py::cast(TimeFunction::BorderType::Cyclic);
}

}  // namespace

PYBIND11_MODULE(base, m) {
  tick::pybind::ensure_numpy_imported();

  m.doc() = "tick.base pybind11 bindings";

  py::class_<A0>(m, "A0")
      .def(py::init<>())
      .def("get_cpp_int", &A0::get_cpp_int)
      .def("set_cpp_int", &A0::set_cpp_int);

  bind_time_function(m);

  m.def("throw_out_of_range", &throw_out_of_range);
  m.def("throw_system_error", &throw_system_error);
  m.def("throw_invalid_argument", &throw_invalid_argument);
  m.def("throw_domain_error", &throw_domain_error);
  m.def("throw_runtime_error", &throw_runtime_error);
  m.def("throw_string", []() {
    try {
      ::throw_string();
    } catch (const std::string &message) {
      throw std::runtime_error(message);
    }
  });

  m.def("standard_normal_cdf", &standard_normal_cdf, py::arg("x"));
  m.def("standard_normal_inv_cdf",
        static_cast<double (*)(double)>(&standard_normal_inv_cdf),
        py::arg("q"));
  m.def("standard_normal_inv_cdf",
        static_cast<void (*)(ArrayDouble &, ArrayDouble &)>(
            &standard_normal_inv_cdf),
        py::arg("q"), py::arg("out"));
}
