#include <memory>

#include <pybind11/pybind11.h>

#include "common/tick_pybind11_arrays.h"
#include "tick/base_model/model.h"
#include "tick/hawkes/model/list_of_realizations/model_hawkes_expkern_leastsq.h"
#include "tick/hawkes/model/list_of_realizations/model_hawkes_expkern_loglik.h"
#include "tick/hawkes/model/list_of_realizations/model_hawkes_sumexpkern_leastsq.h"
#include "tick/hawkes/model/list_of_realizations/model_hawkes_sumexpkern_loglik.h"

namespace py = pybind11;

namespace {

template <typename ModelType>
py::class_<ModelType, std::shared_ptr<ModelType>, ModelDouble> bind_hawkes_common(
    py::module_ &m, const char *name) {
  return py::class_<ModelType, std::shared_ptr<ModelType>, ModelDouble>(m, name)
      .def("get_n_coeffs", &ModelType::get_n_coeffs)
      .def("get_n_nodes", &ModelType::get_n_nodes)
      .def("get_n_total_jumps", &ModelType::get_n_total_jumps)
      .def("get_n_threads", &ModelType::get_n_threads)
      .def("set_n_threads", &ModelType::set_n_threads, py::arg("n_threads"))
      .def("set_data", &ModelType::set_data, py::arg("timestamps_list"),
           py::arg("end_times"))
      .def("incremental_set_data", &ModelType::incremental_set_data,
           py::arg("timestamps"), py::arg("end_time"))
      .def("compute_weights", &ModelType::compute_weights)
      .def("get_end_times", &ModelType::get_end_times);
}

template <typename ModelType, typename... Bases>
void bind_hawkes_leastsq_api(
    py::class_<ModelType, std::shared_ptr<ModelType>, Bases...> &cls) {
  cls.def("loss", &ModelType::loss, py::arg("coeffs"))
      .def("grad", &ModelType::grad, py::arg("coeffs"), py::arg("out"));
}

template <typename ModelType, typename... Bases>
void bind_hawkes_loglik_api(
    py::class_<ModelType, std::shared_ptr<ModelType>, Bases...> &cls) {
  cls.def("loss", &ModelType::loss, py::arg("coeffs"))
      .def("grad", &ModelType::grad, py::arg("coeffs"), py::arg("out"))
      .def("loss_and_grad", &ModelType::loss_and_grad, py::arg("coeffs"),
           py::arg("out"))
      .def("hessian_norm", &ModelType::hessian_norm, py::arg("coeffs"),
           py::arg("vector"))
      .def("hessian", &ModelType::hessian, py::arg("coeffs"),
           py::arg("out"));
}

}  // namespace

PYBIND11_MODULE(hawkes_model, m) {
  tick::pybind::ensure_numpy_imported();
  py::module_::import("tick.base_model.build.base_model");

  m.doc() = "tick.hawkes.model pybind11 bindings";

  auto hawkes_expkern_leastsq =
      bind_hawkes_common<ModelHawkesExpKernLeastSq>(
          m, "ModelHawkesExpKernLeastSq");
  hawkes_expkern_leastsq
      .def(py::init<SArrayDouble2dPtr, int, unsigned int>(),
           py::arg("decays"), py::arg("n_threads") = 1,
           py::arg("approx") = 0)
      .def("set_decays", &ModelHawkesExpKernLeastSq::set_decays,
           py::arg("decays"))
      .def("hessian", &ModelHawkesExpKernLeastSq::hessian, py::arg("out"));
  bind_hawkes_leastsq_api(hawkes_expkern_leastsq);
  tick::pybind::enable_cereal_pickle<ModelHawkesExpKernLeastSq>(
      hawkes_expkern_leastsq);

  auto hawkes_expkern_loglik =
      bind_hawkes_common<ModelHawkesExpKernLogLik>(
          m, "ModelHawkesExpKernLogLik");
  hawkes_expkern_loglik
      .def(py::init<double, int>(), py::arg("decay"),
           py::arg("n_threads") = 1)
      .def("get_decay", &ModelHawkesExpKernLogLik::get_decay)
      .def("set_decay", &ModelHawkesExpKernLogLik::set_decay,
           py::arg("decay"));
  bind_hawkes_loglik_api(hawkes_expkern_loglik);

  auto hawkes_sumexpkern_leastsq =
      bind_hawkes_common<ModelHawkesSumExpKernLeastSq>(
          m, "ModelHawkesSumExpKernLeastSq");
  hawkes_sumexpkern_leastsq
      .def(py::init<const ArrayDouble &, ulong, double, unsigned int,
                    unsigned int>(),
           py::arg("decays"), py::arg("n_baselines"),
           py::arg("period_length"), py::arg("n_threads") = 1,
           py::arg("approx") = 0)
      .def("set_decays", &ModelHawkesSumExpKernLeastSq::set_decays,
           py::arg("decays"))
      .def("get_n_decays", &ModelHawkesSumExpKernLeastSq::get_n_decays)
      .def("get_n_baselines", &ModelHawkesSumExpKernLeastSq::get_n_baselines)
      .def("get_period_length", &ModelHawkesSumExpKernLeastSq::get_period_length)
      .def("set_n_baselines", &ModelHawkesSumExpKernLeastSq::set_n_baselines,
           py::arg("n_baselines"))
      .def("set_period_length",
           &ModelHawkesSumExpKernLeastSq::set_period_length,
           py::arg("period_length"));
  bind_hawkes_leastsq_api(hawkes_sumexpkern_leastsq);
  tick::pybind::enable_cereal_pickle<ModelHawkesSumExpKernLeastSq>(
      hawkes_sumexpkern_leastsq);

  auto hawkes_sumexpkern_loglik =
      bind_hawkes_common<ModelHawkesSumExpKernLogLik>(
          m, "ModelHawkesSumExpKernLogLik");
  hawkes_sumexpkern_loglik
      .def(py::init<const ArrayDouble &, int>(), py::arg("decays"),
           py::arg("n_threads") = 1)
      .def("get_decays", &ModelHawkesSumExpKernLogLik::get_decays)
      .def("set_decays", &ModelHawkesSumExpKernLogLik::set_decays,
           py::arg("decays"))
      .def("get_n_decays", &ModelHawkesSumExpKernLogLik::get_n_decays);
  bind_hawkes_loglik_api(hawkes_sumexpkern_loglik);
}
