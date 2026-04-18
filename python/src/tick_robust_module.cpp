#include <pybind11/pybind11.h>

#include "common/tick_pybind11_arrays.h"
#include "tick/base_model/model_generalized_linear.h"
#include "tick/base_model/model_lipschitz.h"
#include "tick/robust/model_absolute_regression.h"
#include "tick/robust/model_epsilon_insensitive.h"
#include "tick/robust/model_generalized_linear_with_intercepts.h"
#include "tick/robust/model_huber.h"
#include "tick/robust/model_linreg_with_intercepts.h"
#include "tick/robust/model_modified_huber.h"

namespace py = pybind11;

namespace {

template <typename ModelType, typename BaseType>
void bind_glm_model(py::module_ &m, const char *name) {
  auto cls = py::class_<ModelType, py::smart_holder, BaseType>(m,
                                                                         name)
      .def(py::init<SBaseArrayDouble2dPtr, SArrayDoublePtr, bool, int>(),
           py::arg("features"), py::arg("labels"), py::arg("fit_intercept"),
           py::arg("n_threads") = 1)
      .def("compare",
           [](ModelType &self, ModelType &other) {
             return static_cast<bool>(self.compare(other));
           });
  tick::pybind::enable_cereal_pickle<ModelType>(cls);
}

template <typename ModelType, typename BaseType>
void bind_threshold_model(py::module_ &m, const char *name) {
  auto cls = py::class_<ModelType, py::smart_holder, BaseType>(m,
                                                                         name)
      .def(py::init<SBaseArrayDouble2dPtr, SArrayDoublePtr, bool, double, int>(),
           py::arg("features"), py::arg("labels"), py::arg("fit_intercept"),
           py::arg("threshold"), py::arg("n_threads") = 1)
      .def("get_threshold", &ModelType::get_threshold)
      .def("set_threshold", &ModelType::set_threshold)
      .def("compare",
           [](ModelType &self, ModelType &other) {
             return static_cast<bool>(self.compare(other));
           });
  tick::pybind::enable_cereal_pickle<ModelType>(cls);
}

}  // namespace

PYBIND11_MODULE(robust, m) {
  tick::pybind::ensure_numpy_imported();

  m.doc() = "tick.robust pybind11 bindings";

  auto model_glm_with_intercepts =
      py::class_<ModelGeneralizedLinearWithInterceptsDouble, py::smart_holder,
                 ModelGeneralizedLinearDouble>(
          m, "ModelGeneralizedLinearWithInterceptsDouble")
      .def(py::init<SBaseArrayDouble2dPtr, SArrayDoublePtr, bool, int>(),
           py::arg("features"), py::arg("labels"), py::arg("fit_intercept"),
           py::arg("n_threads") = 1)
      .def("compare",
           [](ModelGeneralizedLinearWithInterceptsDouble &self,
              ModelGeneralizedLinearWithInterceptsDouble &other) {
             return static_cast<bool>(self.compare(other));
           });
  tick::pybind::enable_cereal_pickle<ModelGeneralizedLinearWithInterceptsDouble>(
      model_glm_with_intercepts);

  bind_glm_model<ModelAbsoluteRegressionDouble, ModelGeneralizedLinearDouble>(
      m, "ModelAbsoluteRegressionDouble");
  bind_threshold_model<ModelEpsilonInsensitiveDouble,
                       ModelGeneralizedLinearDouble>(
      m, "ModelEpsilonInsensitiveDouble");

  auto huber =
      py::class_<ModelHuberDouble, py::smart_holder,
                 ModelGeneralizedLinearDouble, ModelLipschitzDouble>(
          m, "ModelHuberDouble")
      .def(py::init<SBaseArrayDouble2dPtr, SArrayDoublePtr, bool, double, int>(),
           py::arg("features"), py::arg("labels"), py::arg("fit_intercept"),
           py::arg("threshold"), py::arg("n_threads") = 1)
      .def("get_threshold", &ModelHuberDouble::get_threshold)
      .def("set_threshold", &ModelHuberDouble::set_threshold)
      .def("compare",
           [](ModelHuberDouble &self, ModelHuberDouble &other) {
             return static_cast<bool>(self.compare(other));
           });
  tick::pybind::enable_cereal_pickle<ModelHuberDouble>(huber);

  auto linreg_with_intercepts =
      py::class_<ModelLinRegWithInterceptsDouble, py::smart_holder,
                 ModelGeneralizedLinearWithInterceptsDouble,
                 ModelLipschitzDouble>(m, "ModelLinRegWithInterceptsDouble")
      .def(py::init<SBaseArrayDouble2dPtr, SArrayDoublePtr, bool, int>(),
           py::arg("features"), py::arg("labels"), py::arg("fit_intercept"),
           py::arg("n_threads") = 1)
      .def("compare",
           [](ModelLinRegWithInterceptsDouble &self,
              ModelLinRegWithInterceptsDouble &other) {
             return static_cast<bool>(self.compare(other));
           });
  tick::pybind::enable_cereal_pickle<ModelLinRegWithInterceptsDouble>(
      linreg_with_intercepts);

  auto modified_huber =
      py::class_<ModelModifiedHuberDouble, py::smart_holder,
                 ModelGeneralizedLinearDouble, ModelLipschitzDouble>(
          m, "ModelModifiedHuberDouble")
      .def(py::init<SBaseArrayDouble2dPtr, SArrayDoublePtr, bool, int>(),
           py::arg("features"), py::arg("labels"), py::arg("fit_intercept"),
           py::arg("n_threads") = 1)
      .def("compare",
           [](ModelModifiedHuberDouble &self, ModelModifiedHuberDouble &other) {
             return static_cast<bool>(self.compare(other));
           });
  tick::pybind::enable_cereal_pickle<ModelModifiedHuberDouble>(modified_huber);
}
