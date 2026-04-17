#include <memory>

#include <pybind11/pybind11.h>

#include "common/tick_pybind11_arrays.h"
#include "tick/base_model/model.h"
#include "tick/base_model/model_lipschitz.h"
#include "tick/survival/model_coxreg_partial_lik.h"
#include "tick/survival/model_sccs.h"

namespace py = pybind11;

namespace {

template <typename ModelType, typename BaseType, typename FeaturesPtr,
          typename TimesPtr>
void bind_cox_model(py::module_ &m, const char *name) {
  auto cls = py::class_<ModelType, std::shared_ptr<ModelType>, BaseType>(m,
                                                                         name)
                 .def(py::init<FeaturesPtr, TimesPtr, SArrayUShortPtr>(),
                      py::arg("features"), py::arg("times"),
                      py::arg("censoring"))
                 .def("loss", &ModelType::loss, py::arg("coeffs"))
                 .def("grad", &ModelType::grad, py::arg("coeffs"),
                      py::arg("out"))
                 .def("compare",
                      [](ModelType &self, ModelType &other) {
                        return static_cast<bool>(self.compare(other));
                      });
  tick::pybind::enable_cereal_pickle<ModelType>(cls);
}

}  // namespace

PYBIND11_MODULE(survival, m) {
  tick::pybind::ensure_numpy_imported();
  py::module_::import("tick.base_model.build.base_model");

  m.doc() = "tick.survival pybind11 bindings";

  bind_cox_model<ModelCoxRegPartialLikDouble, ModelDouble, SBaseArrayDouble2dPtr,
                 SArrayDoublePtr>(m, "ModelCoxRegPartialLikDouble");
  bind_cox_model<ModelCoxRegPartialLikFloat, ModelFloat, SBaseArrayFloat2dPtr,
                 SArrayFloatPtr>(m, "ModelCoxRegPartialLikFloat");

  auto model_sccs =
      py::class_<ModelSCCS, std::shared_ptr<ModelSCCS>, ModelLipschitzDouble>(
          m, "ModelSCCS")
          .def(py::init<SBaseArrayDouble2dPtrList1D, SArrayIntPtrList1D,
                        SArrayULongPtr, SArrayULongPtr>(),
               py::arg("features"), py::arg("labels"), py::arg("censoring"),
               py::arg("n_lags"))
          .def("loss", &ModelSCCS::loss, py::arg("coeffs"))
          .def("grad", &ModelSCCS::grad, py::arg("coeffs"), py::arg("out"))
          .def("get_n_coeffs", &ModelSCCS::get_n_coeffs)
          .def("get_epoch_size", &ModelSCCS::get_epoch_size)
          .def("get_rand_max", &ModelSCCS::get_rand_max)
          .def("compare",
               [](ModelSCCS &self, ModelSCCS &other) {
                 return static_cast<bool>(self.compare(other));
               });
  tick::pybind::enable_cereal_pickle<ModelSCCS>(model_sccs);
}
