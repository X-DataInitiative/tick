#include <algorithm>
#include <memory>

#include <pybind11/pybind11.h>

#include "common/tick_pybind11_arrays.h"
#include "tick/hawkes/inference/hawkes_adm4.h"
#include "tick/hawkes/inference/hawkes_basis_kernels.h"
#include "tick/hawkes/inference/hawkes_cumulant.h"
#include "tick/hawkes/inference/hawkes_conditional_law.h"
#include "tick/hawkes/inference/hawkes_em.h"
#include "tick/hawkes/inference/hawkes_sumgaussians.h"
#include "tick/hawkes/model/base/model_hawkes.h"
#include "tick/hawkes/model/base/model_hawkes_list.h"

namespace py = pybind11;

namespace {

VArrayDoublePtr compute_end_times(const SArrayDoublePtrList2D &timestamps_list) {
  const ulong n_realizations = timestamps_list.size();
  auto end_times = VArrayDouble::new_ptr(n_realizations);

  for (ulong r = 0; r < n_realizations; ++r) {
    double end_time = -1.;
    const SArrayDoublePtrList1D &realization = timestamps_list[r];
    for (ulong u = 0; u < realization.size(); ++u) {
      const auto &timestamps = realization[u];
      if (timestamps->size() == 0) {
        continue;
      }
      end_time = std::max(end_time, (*timestamps)[timestamps->size() - 1]);
    }
    if (end_time < 0.) {
      TICK_ERROR("Each realization must contain at least one event");
    }
    (*end_times)[r] = end_time;
  }

  return end_times;
}

void bind_model_hawkes(py::module_ &m) {
  py::class_<ModelHawkes, std::shared_ptr<ModelHawkes>>(m, "ModelHawkes")
      .def("get_n_nodes", &ModelHawkes::get_n_nodes)
      .def("set_n_nodes", &ModelHawkes::set_n_nodes, py::arg("n_nodes"))
      .def("get_n_total_jumps", &ModelHawkes::get_n_total_jumps)
      .def("set_n_threads", &ModelHawkes::set_n_threads,
           py::arg("n_threads"))
      .def("get_n_jumps_per_node", &ModelHawkes::get_n_jumps_per_node);
}

void bind_model_hawkes_list(py::module_ &m) {
  auto cls =
      py::class_<ModelHawkesList, std::shared_ptr<ModelHawkesList>,
                 ModelHawkes>(m, "ModelHawkesList")
          .def(py::init<int, unsigned int>(), py::arg("max_n_threads") = 1,
               py::arg("optimization_level") = 0)
          .def("set_data",
               static_cast<void (ModelHawkesList::*)(
                   const SArrayDoublePtrList2D &, const VArrayDoublePtr)>(
                   &ModelHawkesList::set_data),
               py::arg("timestamps_list"), py::arg("end_times"))
          .def("set_data",
               [](ModelHawkesList &self,
                  const SArrayDoublePtrList2D &timestamps_list) {
                 self.set_data(timestamps_list, compute_end_times(timestamps_list));
               },
               py::arg("timestamps_list"))
          .def("get_n_jumps_per_realization",
               &ModelHawkesList::get_n_jumps_per_realization)
          .def("get_end_times", &ModelHawkesList::get_end_times)
          .def("get_n_threads", &ModelHawkesList::get_n_threads)
          .def("get_timestamps_list", &ModelHawkesList::get_timestamps_list);

  tick::pybind::enable_cereal_pickle<ModelHawkesList>(cls);
}

}  // namespace

PYBIND11_MODULE(hawkes_inference, m) {
  tick::pybind::ensure_numpy_imported();

  m.doc() = "tick.hawkes.inference.build.hawkes_inference pybind11 bindings";

  bind_model_hawkes(m);
  bind_model_hawkes_list(m);

  auto hawkes_adm4 =
      py::class_<HawkesADM4, std::shared_ptr<HawkesADM4>, ModelHawkesList>(
          m, "HawkesADM4")
          .def(py::init<double, double, int, unsigned int>(),
               py::arg("decay"), py::arg("rho"), py::arg("max_n_threads") = 1,
               py::arg("optimization_level") = 0)
          .def("compute_weights", &HawkesADM4::compute_weights)
          .def("solve", &HawkesADM4::solve, py::arg("mu"),
               py::arg("adjacency"), py::arg("z1"), py::arg("z2"),
               py::arg("u1"), py::arg("u2"))
          .def("get_decay", &HawkesADM4::get_decay)
          .def("set_decay", &HawkesADM4::set_decay, py::arg("decay"))
          .def("get_rho", &HawkesADM4::get_rho)
          .def("set_rho", &HawkesADM4::set_rho, py::arg("rho"));

  auto hawkes_em =
      py::class_<HawkesEM, std::shared_ptr<HawkesEM>, ModelHawkesList>(
          m, "HawkesEM")
          .def(py::init<double, ulong, int>(), py::arg("kernel_support"),
               py::arg("kernel_size"), py::arg("max_n_threads") = 1)
          .def(py::init<SArrayDoublePtr, int>(),
               py::arg("kernel_discretization"),
               py::arg("max_n_threads") = 1)
          .def("allocate_weights", &HawkesEM::allocate_weights)
          .def("solve", &HawkesEM::solve, py::arg("mu"), py::arg("kernels"))
          .def("loglikelihood", &HawkesEM::loglikelihood, py::arg("mu"),
               py::arg("kernels"))
          .def("get_kernel_norms", &HawkesEM::get_kernel_norms,
               py::arg("kernels"))
          .def("get_kernel_primitives", &HawkesEM::get_kernel_primitives,
               py::arg("kernels"))
          .def("get_kernel_support", &HawkesEM::get_kernel_support)
          .def("set_kernel_support", &HawkesEM::set_kernel_support,
               py::arg("kernel_support"))
          .def("get_kernel_size", &HawkesEM::get_kernel_size)
          .def("set_kernel_size", &HawkesEM::set_kernel_size,
               py::arg("kernel_size"))
          .def("get_kernel_t0", &HawkesEM::get_kernel_t0)
          .def("get_kernel_dt", &HawkesEM::get_kernel_dt, py::arg("m") = 0)
          .def("get_kernel_fixed_dt", &HawkesEM::get_kernel_fixed_dt)
          .def("get_kernel_discretization",
               &HawkesEM::get_kernel_discretization)
          .def("set_kernel_dt", &HawkesEM::set_kernel_dt,
               py::arg("kernel_dt"))
          .def("set_kernel_discretization",
               &HawkesEM::set_kernel_discretization,
               py::arg("kernel_discretization"))
          .def("init_kernel_time_func", &HawkesEM::init_kernel_time_func,
               py::arg("kernels"))
          .def("set_buffer_variables_for_integral_of_intensity",
               &HawkesEM::set_buffer_variables_for_integral_of_intensity,
               py::arg("mu"), py::arg("kernels"))
          .def("primitive_of_intensity_at_jump_times",
               py::overload_cast<const ulong, ArrayDouble &, ArrayDouble2d &>(
                   &HawkesEM::primitive_of_intensity_at_jump_times),
               py::arg("r_u"), py::arg("mu"), py::arg("kernels"))
          .def("primitive_of_intensity_at_jump_times",
               py::overload_cast<const ulong>(
                   &HawkesEM::primitive_of_intensity_at_jump_times),
               py::arg("r_u"));

  auto hawkes_basis_kernels = py::class_<HawkesBasisKernels,
                                         std::shared_ptr<HawkesBasisKernels>,
                                         ModelHawkesList>(m,
                                                          "HawkesBasisKernels")
                                 .def(py::init<double, ulong, ulong, double,
                                               int>(),
                                      py::arg("kernel_support"),
                                      py::arg("kernel_size"),
                                      py::arg("n_basis"),
                                      py::arg("alpha"),
                                      py::arg("max_n_threads") = 1)
                                 .def("solve", &HawkesBasisKernels::solve,
                                      py::arg("mu"), py::arg("gdm"),
                                      py::arg("auvd"), py::arg("max_iter_gdm"),
                                      py::arg("max_tol_gdm"))
                                 .def("get_kernel_support",
                                      &HawkesBasisKernels::get_kernel_support)
                                 .def("set_kernel_support",
                                      &HawkesBasisKernels::set_kernel_support,
                                      py::arg("kernel_support"))
                                 .def("get_kernel_size",
                                      &HawkesBasisKernels::get_kernel_size)
                                 .def("set_kernel_size",
                                      &HawkesBasisKernels::set_kernel_size,
                                      py::arg("kernel_size"))
                                 .def("get_kernel_dt",
                                      &HawkesBasisKernels::get_kernel_dt)
                                 .def("set_kernel_dt",
                                      &HawkesBasisKernels::set_kernel_dt,
                                      py::arg("kernel_dt"))
                                 .def("get_n_basis",
                                      &HawkesBasisKernels::get_n_basis)
                                 .def("set_n_basis",
                                      &HawkesBasisKernels::set_n_basis,
                                      py::arg("n_basis"))
                                 .def("get_alpha", &HawkesBasisKernels::get_alpha)
                                 .def("set_alpha", &HawkesBasisKernels::set_alpha,
                                      py::arg("alpha"))
                                 .def("get_kernel_discretization",
                                      &HawkesBasisKernels::get_kernel_discretization);

  auto hawkes_sum_gaussians =
      py::class_<HawkesSumGaussians, std::shared_ptr<HawkesSumGaussians>,
                 ModelHawkesList>(m, "HawkesSumGaussians")
          .def(py::init<ulong, double, double, double, double, ulong, int,
                        unsigned int>(),
               py::arg("n_gaussians"), py::arg("max_mean_gaussian"),
               py::arg("step_size"), py::arg("strength_lasso"),
               py::arg("strength_grouplasso"), py::arg("em_max_iter"),
               py::arg("max_n_threads") = 1,
               py::arg("optimization_level") = 0)
          .def("compute_weights", &HawkesSumGaussians::compute_weights)
          .def("solve", &HawkesSumGaussians::solve, py::arg("mu"),
               py::arg("amplitudes"))
          .def("get_n_gaussians", &HawkesSumGaussians::get_n_gaussians)
          .def("set_n_gaussians", &HawkesSumGaussians::set_n_gaussians,
               py::arg("n_gaussians"))
          .def("get_em_max_iter", &HawkesSumGaussians::get_em_max_iter)
          .def("set_em_max_iter", &HawkesSumGaussians::set_em_max_iter,
               py::arg("em_max_iter"))
          .def("get_max_mean_gaussian",
               &HawkesSumGaussians::get_max_mean_gaussian)
          .def("set_max_mean_gaussian",
               &HawkesSumGaussians::set_max_mean_gaussian,
               py::arg("max_mean_gaussian"))
          .def("get_step_size", &HawkesSumGaussians::get_step_size)
          .def("set_step_size", &HawkesSumGaussians::set_step_size,
               py::arg("step_size"))
          .def("get_strength_lasso", &HawkesSumGaussians::get_strength_lasso)
          .def("set_strength_lasso", &HawkesSumGaussians::set_strength_lasso,
               py::arg("strength_lasso"))
          .def("get_strength_grouplasso",
               &HawkesSumGaussians::get_strength_grouplasso)
          .def("set_strength_grouplasso",
               &HawkesSumGaussians::set_strength_grouplasso,
               py::arg("strength_grouplasso"));

  auto hawkes_cumulant =
      py::class_<HawkesCumulant, std::shared_ptr<HawkesCumulant>,
                 ModelHawkesList>(m, "HawkesCumulant")
          .def(py::init<double>(), py::arg("integration_support"))
          .def("compute_A_and_I_ij", &HawkesCumulant::compute_A_and_I_ij,
               py::arg("r"), py::arg("i"), py::arg("j"),
               py::arg("mean_intensity_j"))
          .def("compute_E_ijk", &HawkesCumulant::compute_E_ijk,
               py::arg("r"), py::arg("i"), py::arg("j"), py::arg("k"),
               py::arg("mean_intensity_i"), py::arg("mean_intensity_j"),
               py::arg("J_ij"))
          .def("get_integration_support",
               &HawkesCumulant::get_integration_support)
          .def("set_integration_support",
               &HawkesCumulant::set_integration_support,
               py::arg("integration_support"))
          .def("get_are_cumulants_ready",
               &HawkesCumulant::get_are_cumulants_ready)
          .def("set_are_cumulants_ready",
               &HawkesCumulant::set_are_cumulants_ready,
               py::arg("are_cumulants_ready"));

  py::class_<HawkesTheoreticalCumulant,
             std::shared_ptr<HawkesTheoreticalCumulant>>(m,
                                                         "HawkesTheoreticalCumulant")
      .def(py::init<int>(), py::arg("dim"))
      .def("get_dimension", &HawkesTheoreticalCumulant::get_dimension)
      .def("set_baseline", &HawkesTheoreticalCumulant::set_baseline,
           py::arg("mu"))
      .def("get_baseline", &HawkesTheoreticalCumulant::get_baseline)
      .def("set_g_geom", &HawkesTheoreticalCumulant::set_g_geom,
           py::arg("g_geom"))
      .def("get_g_geom", &HawkesTheoreticalCumulant::get_g_geom)
      .def("compute_mean_intensity",
           &HawkesTheoreticalCumulant::compute_mean_intensity)
      .def("compute_covariance", &HawkesTheoreticalCumulant::compute_covariance)
      .def("compute_skewness", &HawkesTheoreticalCumulant::compute_skewness)
      .def("compute_cumulants", &HawkesTheoreticalCumulant::compute_cumulants)
      .def("mean_intensity", &HawkesTheoreticalCumulant::mean_intensity)
      .def("covariance", &HawkesTheoreticalCumulant::covariance)
      .def("skewness", &HawkesTheoreticalCumulant::skewness);

  m.def("PointProcessCondLaw", &PointProcessCondLaw, py::arg("y_time"),
        py::arg("z_time"), py::arg("z_mark"), py::arg("lags"),
        py::arg("zmin"), py::arg("zmax"), py::arg("y_T"),
        py::arg("y_lambda"), py::arg("res_X"), py::arg("res_Y"));
}
