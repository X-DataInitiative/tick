#include <pybind11/pybind11.h>

#include "common/tick_pybind11_arrays.h"
#include "tick/base_model/model.h"
#include "tick/base_model/model_generalized_linear.h"
#include "tick/base_model/model_labels_features.h"
#include "tick/base_model/model_lipschitz.h"

namespace py = pybind11;

namespace {

template <typename ModelType, typename Scalar>
void bind_model_base(py::module_ &m, const char *name) {
  py::class_<ModelType, std::shared_ptr<ModelType>>(m, name)
      .def("grad", &ModelType::grad, py::arg("coeffs"), py::arg("out"))
      .def("loss", &ModelType::loss, py::arg("coeffs"))
      .def("get_epoch_size", &ModelType::get_epoch_size)
      .def("is_sparse", &ModelType::is_sparse);
}

template <typename ModelType, typename BaseType>
void bind_model_labels_features(py::module_ &m, const char *name) {
  py::class_<ModelType, std::shared_ptr<ModelType>, BaseType>(m, name)
      .def("get_n_samples", &ModelType::get_n_samples)
      .def("get_n_features", &ModelType::get_n_features);
}

template <typename ModelType, typename BaseType>
void bind_model_lipschitz(py::module_ &m, const char *name) {
  py::class_<ModelType, std::shared_ptr<ModelType>, BaseType>(m, name)
      .def("get_lip_max", &ModelType::get_lip_max)
      .def("get_lip_mean", &ModelType::get_lip_mean);
}

template <typename ModelType, typename BaseType, typename Array2dPtr,
          typename ArrayPtr, typename Scalar>
void bind_model_generalized_linear(py::module_ &m, const char *name) {
  py::class_<ModelType, std::shared_ptr<ModelType>, BaseType>(m, name)
      .def(py::init<Array2dPtr, ArrayPtr, bool, int>(), py::arg("features"),
           py::arg("labels"), py::arg("fit_intercept"),
           py::arg("n_threads") = 1)
      .def("get_n_coeffs", &ModelType::get_n_coeffs)
      .def("set_fit_intercept", &ModelType::set_fit_intercept,
           py::arg("fit_intercept"))
      .def("get_fit_intercept", &ModelType::get_fit_intercept)
      .def("sdca_primal_dual_relation", &ModelType::sdca_primal_dual_relation,
           py::arg("l_l2sq"), py::arg("dual_vector"),
           py::arg("out_primal_vector"));
}

}  // namespace

PYBIND11_MODULE(base_model, m) {
  tick::pybind::ensure_numpy_imported();

  m.doc() = "tick.base_model pybind11 bindings";

  bind_model_base<ModelDouble, double>(m, "ModelDouble");
  bind_model_base<ModelFloat, float>(m, "ModelFloat");
  m.attr("Model") = m.attr("ModelDouble");

  bind_model_labels_features<ModelLabelsFeaturesDouble, ModelDouble>(
      m, "ModelLabelsFeaturesDouble");
  bind_model_labels_features<ModelLabelsFeaturesFloat, ModelFloat>(
      m, "ModelLabelsFeaturesFloat");

  bind_model_lipschitz<ModelLipschitzDouble, ModelDouble>(
      m, "ModelLipschitzDouble");
  bind_model_lipschitz<ModelLipschitzFloat, ModelFloat>(
      m, "ModelLipschitzFloat");
  m.attr("ModelLipschitz") = m.attr("ModelLipschitzDouble");

  bind_model_generalized_linear<ModelGeneralizedLinearDouble,
                                ModelLabelsFeaturesDouble, SBaseArrayDouble2dPtr,
                                SArrayDoublePtr, double>(
      m, "ModelGeneralizedLinearDouble");
  bind_model_generalized_linear<ModelGeneralizedLinearFloat,
                                ModelLabelsFeaturesFloat, SBaseArrayFloat2dPtr,
                                SArrayFloatPtr, float>(
      m, "ModelGeneralizedLinearFloat");
}
