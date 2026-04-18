#include <pybind11/pybind11.h>

#include "common/tick_pybind11_arrays.h"
#include "tick/base_model/model_generalized_linear.h"
#include "tick/base_model/model_lipschitz.h"
#include "tick/linear_model/model_hinge.h"
#include "tick/linear_model/model_linreg.h"
#include "tick/linear_model/model_logreg.h"
#include "tick/linear_model/model_poisreg.h"
#include "tick/linear_model/model_quadratic_hinge.h"
#include "tick/linear_model/model_smoothed_hinge.h"

namespace py = pybind11;

namespace {

template <typename ModelType, typename BaseGeneralizedLinear,
          typename BaseLipschitz, typename Array2dPtr, typename ArrayPtr,
          typename Scalar>
void bind_model_glm(py::module_ &m, const char *name) {
  auto cls = py::class_<ModelType, py::smart_holder,
                        BaseGeneralizedLinear, BaseLipschitz>(m, name)
                 .def(py::init<Array2dPtr, ArrayPtr, bool, int>(), py::arg("features"),
                      py::arg("labels"), py::arg("fit_intercept"),
                      py::arg("n_threads") = 1)
                 .def("compare",
                      [](ModelType &self, ModelType &other) {
                        return static_cast<bool>(self.compare(other));
                      });
  tick::pybind::enable_cereal_pickle<ModelType>(cls);
}

template <typename Scalar>
void bind_sigmoid(py::class_<TModelLogReg<Scalar, Scalar>, py::smart_holder,
                             TModelGeneralizedLinear<Scalar, Scalar>,
                             TModelLipschitz<Scalar, Scalar>> &cls) {
  cls.def_static(
      "sigmoid",
      [](const Array<Scalar> &x, Array<Scalar> &out) {
        TModelLogReg<Scalar, Scalar>::sigmoid(x, out);
      },
      py::arg("x"), py::arg("out"));
}

template <typename ModelType, typename BaseGeneralizedLinear, typename Array2dPtr,
          typename ArrayPtr>
void bind_model_poisreg(py::module_ &m, const char *name) {
  auto cls = py::class_<ModelType, py::smart_holder,
                        BaseGeneralizedLinear>(
      m, name)
      .def(py::init<Array2dPtr, ArrayPtr, LinkType, bool, int>(),
           py::arg("features"), py::arg("labels"), py::arg("link_type"),
           py::arg("fit_intercept"), py::arg("n_threads") = 1)
      .def("get_link_type", &ModelType::get_link_type)
      .def("set_link_type", &ModelType::set_link_type)
      .def("compare",
           [](ModelType &self, ModelType &other) {
             return static_cast<bool>(self.compare(other));
           });
  tick::pybind::enable_cereal_pickle<ModelType>(cls);
}

template <typename ModelType, typename BaseGeneralizedLinear, typename Array2dPtr,
          typename ArrayPtr>
void bind_model_hinge(py::module_ &m, const char *name) {
  auto cls = py::class_<ModelType, py::smart_holder,
                        BaseGeneralizedLinear>(m, name)
                 .def(py::init<Array2dPtr, ArrayPtr, bool, int>(),
                      py::arg("features"), py::arg("labels"),
                      py::arg("fit_intercept"), py::arg("n_threads") = 1)
                 .def("compare",
                      [](ModelType &self, ModelType &other) {
                        return static_cast<bool>(self.compare(other));
                      });
  tick::pybind::enable_cereal_pickle<ModelType>(cls);
}

template <typename ModelType, typename BaseGeneralizedLinear,
          typename BaseLipschitz, typename Array2dPtr, typename ArrayPtr,
          typename Scalar>
void bind_model_smoothed_hinge(py::module_ &m, const char *name) {
  auto cls = py::class_<ModelType, py::smart_holder,
                        BaseGeneralizedLinear, BaseLipschitz>(m, name)
                 .def(py::init<Array2dPtr, ArrayPtr, bool, Scalar, int>(),
                      py::arg("features"), py::arg("labels"),
                      py::arg("fit_intercept"), py::arg("smoothness"),
                      py::arg("n_threads") = 1)
                 .def("get_smoothness", &ModelType::get_smoothness)
                 .def("set_smoothness", &ModelType::set_smoothness)
                 .def("compare",
                      [](ModelType &self, ModelType &other) {
                        return static_cast<bool>(self.compare(other));
                      });
  tick::pybind::enable_cereal_pickle<ModelType>(cls);
}

}  // namespace

PYBIND11_MODULE(linear_model, m) {
  tick::pybind::ensure_numpy_imported();

  m.doc() = "tick.linear_model pybind11 bindings";

  py::enum_<LinkType>(m, "LinkType")
      .value("LinkType_identity", LinkType::identity)
      .value("LinkType_exponential", LinkType::exponential)
      .export_values();

  bind_model_glm<ModelLinRegDouble, ModelGeneralizedLinearDouble,
                 ModelLipschitzDouble, SBaseArrayDouble2dPtr, SArrayDoublePtr,
                 double>(m, "ModelLinRegDouble");
  bind_model_glm<ModelLinRegFloat, ModelGeneralizedLinearFloat,
                 ModelLipschitzFloat, SBaseArrayFloat2dPtr, SArrayFloatPtr,
                 float>(m, "ModelLinRegFloat");

  auto logreg_double =
      py::class_<ModelLogRegDouble, py::smart_holder,
                 ModelGeneralizedLinearDouble, ModelLipschitzDouble>(
          m, "ModelLogRegDouble");
  logreg_double
      .def(py::init<SBaseArrayDouble2dPtr, SArrayDoublePtr, bool, int>(),
           py::arg("features"), py::arg("labels"),
           py::arg("fit_intercept"), py::arg("n_threads") = 1)
      .def("compare",
           [](ModelLogRegDouble &self, ModelLogRegDouble &other) {
             return static_cast<bool>(self.compare(other));
           });
  bind_sigmoid<double>(logreg_double);
  tick::pybind::enable_cereal_pickle<ModelLogRegDouble>(logreg_double);

  auto logreg_float = py::class_<ModelLogRegFloat, py::smart_holder,
                                 ModelGeneralizedLinearFloat,
                                 ModelLipschitzFloat>(m, "ModelLogRegFloat");
  logreg_float
      .def(py::init<SBaseArrayFloat2dPtr, SArrayFloatPtr, bool, int>(),
           py::arg("features"), py::arg("labels"),
           py::arg("fit_intercept"), py::arg("n_threads") = 1)
      .def("compare",
           [](ModelLogRegFloat &self, ModelLogRegFloat &other) {
             return static_cast<bool>(self.compare(other));
           });
  bind_sigmoid<float>(logreg_float);
  tick::pybind::enable_cereal_pickle<ModelLogRegFloat>(logreg_float);

  bind_model_poisreg<ModelPoisRegDouble, ModelGeneralizedLinearDouble,
                     SBaseArrayDouble2dPtr, SArrayDoublePtr>(
      m, "ModelPoisRegDouble");
  bind_model_poisreg<ModelPoisRegFloat, ModelGeneralizedLinearFloat,
                     SBaseArrayFloat2dPtr, SArrayFloatPtr>(
      m, "ModelPoisRegFloat");

  bind_model_hinge<ModelHingeDouble, ModelGeneralizedLinearDouble,
                   SBaseArrayDouble2dPtr, SArrayDoublePtr>(m,
                                                           "ModelHingeDouble");
  bind_model_hinge<ModelHingeFloat, ModelGeneralizedLinearFloat,
                   SBaseArrayFloat2dPtr, SArrayFloatPtr>(m, "ModelHingeFloat");

  bind_model_smoothed_hinge<ModelSmoothedHingeDouble,
                            ModelGeneralizedLinearDouble, ModelLipschitzDouble,
                            SBaseArrayDouble2dPtr, SArrayDoublePtr, double>(
      m, "ModelSmoothedHingeDouble");
  bind_model_smoothed_hinge<ModelSmoothedHingeFloat,
                            ModelGeneralizedLinearFloat, ModelLipschitzFloat,
                            SBaseArrayFloat2dPtr, SArrayFloatPtr, float>(
      m, "ModelSmoothedHingeFloat");

  bind_model_glm<ModelQuadraticHingeDouble, ModelGeneralizedLinearDouble,
                 ModelLipschitzDouble, SBaseArrayDouble2dPtr, SArrayDoublePtr,
                 double>(m, "ModelQuadraticHingeDouble");
  bind_model_glm<ModelQuadraticHingeFloat, ModelGeneralizedLinearFloat,
                 ModelLipschitzFloat, SBaseArrayFloat2dPtr, SArrayFloatPtr,
                 float>(m, "ModelQuadraticHingeFloat");
}
