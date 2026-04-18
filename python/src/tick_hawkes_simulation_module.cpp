#include <memory>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "common/tick_pybind11_arrays.h"
#include "tick/base/time_func.h"
#include "tick/hawkes/simulation/hawkes_kernels/hawkes_kernel.h"
#include "tick/hawkes/simulation/hawkes_kernels/hawkes_kernel_0.h"
#include "tick/hawkes/simulation/hawkes_kernels/hawkes_kernel_exp.h"
#include "tick/hawkes/simulation/hawkes_kernels/hawkes_kernel_power_law.h"
#include "tick/hawkes/simulation/hawkes_kernels/hawkes_kernel_sum_exp.h"
#include "tick/hawkes/simulation/hawkes_kernels/hawkes_kernel_time_func.h"
#include "tick/hawkes/simulation/simu_hawkes.h"
#include "tick/hawkes/simulation/simu_inhomogeneous_poisson.h"
#include "tick/hawkes/simulation/simu_point_process.h"
#include "tick/hawkes/simulation/simu_poisson_process.h"

namespace py = pybind11;

namespace {

void bind_point_process(py::module_ &m) {
  py::class_<PP, py::smart_holder>(m, "PP")
      .def(py::init<unsigned int, int>(), py::arg("n_nodes"),
           py::arg("seed") = -1)
      .def("activate_itr", &PP::activate_itr, py::arg("dt"))
      .def("simulate", py::overload_cast<double>(&PP::simulate),
           py::arg("run_time"))
      .def("simulate", py::overload_cast<ulong>(&PP::simulate),
           py::arg("n_points"))
      .def("simulate", py::overload_cast<double, ulong>(&PP::simulate),
           py::arg("run_time"), py::arg("n_points"))
      .def("reset", &PP::reset)
      .def("itr_on", &PP::itr_on)
      .def("get_time", &PP::get_time)
      .def("get_n_nodes", &PP::get_n_nodes)
      .def("get_seed", &PP::get_seed)
      .def("get_n_total_jumps", &PP::get_n_total_jumps)
      .def("get_itr", &PP::get_itr)
      .def("get_itr_times", &PP::get_itr_times)
      .def("get_itr_step", &PP::get_itr_step)
      .def("get_ctr", &PP::get_ctr)
      .def("store_compensator_values", &PP::store_compensator_values)
      .def("evaluate_compensator", &PP::evaluate_compensator, py::arg("node"),
           py::arg("time"))
      .def("reseed_random_generator", &PP::reseed_random_generator,
           py::arg("seed"))
      .def("get_timestamps", &PP::get_timestamps)
      .def("set_timestamps", &PP::set_timestamps, py::arg("timestamps"),
           py::arg("end_time"))
      .def("get_threshold_negative_intensity",
           &PP::get_threshold_negative_intensity)
      .def("set_threshold_negative_intensity",
           &PP::set_threshold_negative_intensity,
           py::arg("threshold_negative_intensity"));
}

void bind_hawkes_kernels(py::module_ &m) {
  py::class_<HawkesKernel, py::smart_holder>(m, "HawkesKernel")
      .def(py::init<double>(), py::arg("support") = 0.0)
      .def("is_zero", &HawkesKernel::is_zero)
      .def("get_support", &HawkesKernel::get_support)
      .def("get_plot_support", &HawkesKernel::get_plot_support)
      .def("get_value", &HawkesKernel::get_value, py::arg("x"))
      .def("get_values", &HawkesKernel::get_values, py::arg("t_values"))
      .def("get_primitive_value",
           static_cast<double (HawkesKernel::*)(double)>(
               &HawkesKernel::get_primitive_value),
           py::arg("t"))
      .def("get_primitive_value",
           static_cast<double (HawkesKernel::*)(double, double)>(
               &HawkesKernel::get_primitive_value),
           py::arg("s"), py::arg("t"))
      .def("get_primitive_values", &HawkesKernel::get_primitive_values,
           py::arg("t_values"))
      .def("get_norm", &HawkesKernel::get_norm, py::arg("nsteps") = 10000);

  auto hawkes_kernel_exp =
      py::class_<HawkesKernelExp, py::smart_holder,
                 HawkesKernel>(
          m, "HawkesKernelExp")
          .def(py::init<double, double>(), py::arg("intensity"),
               py::arg("decay"))
          .def_static("set_fast_exp", &HawkesKernelExp::set_fast_exp,
                      py::arg("flag"))
          .def_static("get_fast_exp", &HawkesKernelExp::get_fast_exp)
          .def("get_intensity", &HawkesKernelExp::get_intensity)
          .def("get_decay", &HawkesKernelExp::get_decay);
  tick::pybind::enable_cereal_pickle<HawkesKernelExp>(hawkes_kernel_exp);

  auto hawkes_kernel_sum_exp =
      py::class_<HawkesKernelSumExp, py::smart_holder,
                 HawkesKernel>(
          m, "HawkesKernelSumExp")
          .def(py::init<const ArrayDouble &, const ArrayDouble &>(),
               py::arg("intensities"), py::arg("decays"))
          .def(py::init<>())
          .def_static("set_fast_exp", &HawkesKernelSumExp::set_fast_exp,
                      py::arg("flag"))
          .def_static("get_fast_exp", &HawkesKernelSumExp::get_fast_exp)
          .def("get_intensities", &HawkesKernelSumExp::get_intensities)
          .def("get_decays", &HawkesKernelSumExp::get_decays)
          .def("get_n_decays", &HawkesKernelSumExp::get_n_decays)
          .def("get_convolution",
               [](HawkesKernelSumExp &self, double time,
                  const ArrayDouble &timestamps) {
                 return self.get_convolution(time, timestamps, nullptr);
               },
               py::arg("time"), py::arg("timestamps"));
  tick::pybind::enable_cereal_pickle<HawkesKernelSumExp>(
      hawkes_kernel_sum_exp);

  auto hawkes_kernel_power_law =
      py::class_<HawkesKernelPowerLaw, py::smart_holder,
                 HawkesKernel>(
          m, "HawkesKernelPowerLaw")
          .def(py::init<double, double, double, double, double>(),
               py::arg("multiplier"), py::arg("cutoff"), py::arg("exponent"),
               py::arg("support") = -1.0, py::arg("error") = 1e-5)
          .def("get_multiplier", &HawkesKernelPowerLaw::get_multiplier)
          .def("get_exponent", &HawkesKernelPowerLaw::get_exponent)
          .def("get_cutoff", &HawkesKernelPowerLaw::get_cutoff);
  tick::pybind::enable_cereal_pickle<HawkesKernelPowerLaw>(
      hawkes_kernel_power_law);

  auto hawkes_kernel_time_func =
      py::class_<HawkesKernelTimeFunc, py::smart_holder,
                 HawkesKernel>(
          m, "HawkesKernelTimeFunc")
          .def(py::init<const TimeFunction &>(), py::arg("time_function"))
          .def(py::init<const ArrayDouble &, const ArrayDouble &>(),
               py::arg("t_axis"), py::arg("y_axis"))
          .def(py::init<>())
          .def("get_time_function", &HawkesKernelTimeFunc::get_time_function);
  tick::pybind::enable_cereal_pickle<HawkesKernelTimeFunc>(
      hawkes_kernel_time_func);

  auto hawkes_kernel_0 =
      py::class_<HawkesKernel0, py::smart_holder, HawkesKernel>(
          m, "HawkesKernel0")
          .def(py::init<>());
  tick::pybind::enable_cereal_pickle<HawkesKernel0>(hawkes_kernel_0);
}

void bind_poisson(py::module_ &m) {
  py::class_<Poisson, py::smart_holder, PP>(m, "Poisson")
      .def(py::init<double, int>(), py::arg("intensity"), py::arg("seed") = -1)
      .def(py::init<SArrayDoublePtr, int>(), py::arg("intensities"),
           py::arg("seed") = -1)
      .def("get_intensities", &Poisson::get_intensities);
}

void bind_inhomogeneous_poisson(py::module_ &m) {
  py::class_<InhomogeneousPoisson, py::smart_holder, PP>(
      m, "InhomogeneousPoisson")
      .def(py::init<const TimeFunction &, int>(), py::arg("intensity_function"),
           py::arg("seed") = -1)
      .def(py::init<const std::vector<TimeFunction> &, int>(),
           py::arg("intensity_functions"), py::arg("seed") = -1)
      .def("intensity_value", &InhomogeneousPoisson::intensity_value,
           py::arg("dimension"), py::arg("times_values"));
}

void bind_hawkes(py::module_ &m) {
  auto hawkes = py::class_<Hawkes, py::smart_holder, PP>(m, "Hawkes")
                    .def(py::init<unsigned int, int>(), py::arg("dimension"),
                         py::arg("seed") = -1)
                    .def("set_kernel", &Hawkes::set_kernel, py::arg("i"),
                         py::arg("j"), py::arg("kernel"))
                    .def("set_baseline",
                         py::overload_cast<unsigned int, double>(
                             &Hawkes::set_baseline),
                         py::arg("i"), py::arg("baseline"))
                    .def("set_baseline",
                         py::overload_cast<unsigned int, ArrayDouble &,
                                           ArrayDouble &>(
                             &Hawkes::set_baseline),
                         py::arg("i"), py::arg("times"),
                         py::arg("values"))
                    .def("set_baseline",
                         py::overload_cast<unsigned int, TimeFunction>(
                             &Hawkes::set_baseline),
                         py::arg("i"), py::arg("time_function"))
                    .def("get_baseline",
                         py::overload_cast<unsigned int, ArrayDouble &>(
                             &Hawkes::get_baseline),
                         py::arg("i"), py::arg("t"))
                    .def("get_baseline",
                         py::overload_cast<unsigned int, double>(
                             &Hawkes::get_baseline),
                         py::arg("i"), py::arg("t"))
                    .def("evaluate_compensator", &Hawkes::evaluate_compensator,
                         py::arg("node"), py::arg("time"))
                    .def(
                        py::pickle(
                            [](const Hawkes &value) {
                              return py::make_tuple(tick::object_to_string(
                                  const_cast<Hawkes *>(&value)));
                            },
                            [](py::tuple state) {
                              if (state.size() != 1) {
                                throw std::runtime_error(
                                    "Invalid Hawkes pickle state");
                              }
                              auto value = std::make_shared<Hawkes>(1);
                              tick::object_from_string(
                                  value.get(), state[0].cast<std::string>());
                              return value;
                            }));
}

}  // namespace

PYBIND11_MODULE(hawkes_simulation, m) {
  tick::pybind::ensure_numpy_imported();

  m.doc() = "tick.hawkes.simulation pybind11 bindings";

  bind_point_process(m);
  bind_hawkes_kernels(m);
  bind_poisson(m);
  bind_inhomogeneous_poisson(m);
  bind_hawkes(m);
}
