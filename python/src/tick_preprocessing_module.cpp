#include <memory>
#include <vector>

#include <pybind11/pybind11.h>

#include "common/tick_pybind11_arrays.h"
#include "tick/preprocessing/longitudinal_features_lagger.h"
#include "tick/preprocessing/sparse_longitudinal_features_product.h"

namespace py = pybind11;

PYBIND11_MODULE(preprocessing, m) {
  tick::pybind::ensure_numpy_imported();

  m.doc() = "tick.preprocessing pybind11 bindings";

  auto load_features = [](py::iterable features) {
    SBaseArrayDouble2dPtrList1D converted;
    for (py::handle item : features) {
      converted.push_back(item.cast<SBaseArrayDouble2dPtr>());
    }
    return converted;
  };

  auto lagger =
      py::class_<LongitudinalFeaturesLagger, py::smart_holder>(
          m, "LongitudinalFeaturesLagger")
          .def(py::init([&load_features](py::iterable features,
                                         SArrayULongPtr n_lags) {
                 return std::make_shared<LongitudinalFeaturesLagger>(
                     load_features(features), n_lags);
               }),
               py::arg("features"), py::arg("n_lags"))
          .def("dense_lag_preprocessor",
               &LongitudinalFeaturesLagger::dense_lag_preprocessor,
               py::arg("features"), py::arg("out"), py::arg("censoring"))
          .def("sparse_lag_preprocessor",
               &LongitudinalFeaturesLagger::sparse_lag_preprocessor,
               py::arg("row"), py::arg("col"), py::arg("data"),
               py::arg("out_row"), py::arg("out_col"), py::arg("out_data"),
               py::arg("censoring"))
          .def(py::pickle(
              [](const LongitudinalFeaturesLagger &value) {
                return py::make_tuple(
                    tick::object_to_string(
                        const_cast<LongitudinalFeaturesLagger *>(&value)));
              },
              [](py::tuple state) {
                if (state.size() != 1) {
                  throw std::runtime_error("Invalid pickle state");
                }

                auto dummy_feature = ArrayDouble2d(1, 1);
                auto dummy_features = SBaseArrayDouble2dPtrList1D{
                    SArrayDouble2d::new_ptr(dummy_feature)};
                auto dummy_n_lags = ArrayULong(1);
                dummy_n_lags[0] = 0;

                auto value = std::make_shared<LongitudinalFeaturesLagger>(
                    dummy_features, SArrayULong::new_ptr(dummy_n_lags));
                tick::object_from_string(value.get(),
                                         state[0].cast<std::string>());
                return value;
              }));

  auto product =
      py::class_<SparseLongitudinalFeaturesProduct, py::smart_holder>(
          m, "SparseLongitudinalFeaturesProduct")
          .def(py::init([&load_features](py::iterable features) {
                 return std::make_shared<SparseLongitudinalFeaturesProduct>(
                     load_features(features));
               }),
               py::arg("features"))
          .def("sparse_features_product",
               &SparseLongitudinalFeaturesProduct::sparse_features_product,
               py::arg("row"), py::arg("col"), py::arg("data"),
               py::arg("out_row"), py::arg("out_col"), py::arg("out_data"))
          .def(py::pickle(
              [](const SparseLongitudinalFeaturesProduct &value) {
                return py::make_tuple(
                    tick::object_to_string(
                        const_cast<SparseLongitudinalFeaturesProduct *>(
                            &value)));
              },
              [](py::tuple state) {
                if (state.size() != 1) {
                  throw std::runtime_error("Invalid pickle state");
                }

                auto dummy_feature = ArrayDouble2d(1, 1);
                auto dummy_features = SBaseArrayDouble2dPtrList1D{
                    SArrayDouble2d::new_ptr(dummy_feature)};
                auto value =
                    std::make_shared<SparseLongitudinalFeaturesProduct>(
                        dummy_features);
                tick::object_from_string(value.get(),
                                         state[0].cast<std::string>());
                return value;
              }));
}
