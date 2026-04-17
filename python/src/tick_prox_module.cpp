#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "common/tick_pybind11_arrays.h"
#include "tick/prox/prox.h"
#include "tick/prox/prox_binarsity.h"
#include "tick/prox/prox_elasticnet.h"
#include "tick/prox/prox_equality.h"
#include "tick/prox/prox_group_l1.h"
#include "tick/prox/prox_l1.h"
#include "tick/prox/prox_l1w.h"
#include "tick/prox/prox_l2.h"
#include "tick/prox/prox_l2sq.h"
#include "tick/prox/prox_multi.h"
#include "tick/prox/prox_positive.h"
#include "tick/prox/prox_separable.h"
#include "tick/prox/prox_slope.h"
#include "tick/prox/prox_sorted_l1.h"
#include "tick/prox/prox_tv.h"
#include "tick/prox/prox_zero.h"

namespace py = pybind11;

namespace {

template <typename ProxType, typename Scalar>
void bind_prox_base(py::module_ &m, const char *name) {
  auto cls = py::class_<ProxType, py::smart_holder>(m, name)
      .def("call",
           [](ProxType &self, const Array<Scalar> &coeffs, Scalar step,
              Array<Scalar> &out) { self.call(coeffs, step, out); },
           py::arg("coeffs"), py::arg("step"), py::arg("out"))
      .def("value",
           py::overload_cast<const Array<Scalar> &>(&ProxType::value),
           py::arg("coeffs"))
      .def("get_strength", &ProxType::get_strength)
      .def("set_strength", &ProxType::set_strength, py::arg("strength"))
      .def("get_start", &ProxType::get_start)
      .def("get_end", &ProxType::get_end)
      .def("set_start_end", &ProxType::set_start_end, py::arg("start"),
           py::arg("end"))
      .def("get_positive", &ProxType::get_positive)
      .def("set_positive", &ProxType::set_positive, py::arg("positive"));
  tick::pybind::enable_cereal_pickle<ProxType>(cls);
}

template <typename ProxType, typename BaseType, typename Scalar>
auto bind_basic_prox(py::module_ &m, const char *name) {
  return py::class_<ProxType, py::smart_holder, BaseType>(m, name)
      .def(py::init<Scalar, bool>(), py::arg("strength"),
           py::arg("positive"))
      .def(py::init<Scalar, ulong, ulong, bool>(), py::arg("strength"),
           py::arg("start"), py::arg("end"), py::arg("positive"))
      .def("compare",
           [](ProxType &self, ProxType &other) {
             return static_cast<bool>(self.compare(other));
           })
      .def("__eq__",
           [](ProxType &self, ProxType &other) {
             return static_cast<bool>(self.compare(other));
           });
}

template <typename ClassType, typename ProxType, typename Scalar>
void bind_array_step_call(ClassType &cls) {
  cls.def(
      "call",
      [](ProxType &self, const Array<Scalar> &coeffs, Scalar step,
         Array<Scalar> &out) {
        static_cast<TProx<Scalar, Scalar> &>(self).call(coeffs, step, out);
      },
      py::arg("coeffs"), py::arg("step"), py::arg("out"));
  cls.def(
      "call",
      [](ProxType &self, const Array<Scalar> &coeffs, const Array<Scalar> &step,
         Array<Scalar> &out) {
        static_cast<TProxSeparable<Scalar, Scalar> &>(self).call(coeffs, step,
                                                                 out);
      },
      py::arg("coeffs"), py::arg("step"), py::arg("out"));
}

template <typename ProxType, typename BaseType, typename Scalar>
void bind_group_prox(py::module_ &m, const char *name) {
  py::class_<ProxType, py::smart_holder, BaseType>(m, name)
      .def(py::init<Scalar, SArrayULongPtr, SArrayULongPtr, bool>(),
           py::arg("strength"), py::arg("blocks_start"),
           py::arg("blocks_length"), py::arg("positive"))
      .def(py::init<Scalar, SArrayULongPtr, SArrayULongPtr, ulong, ulong,
                    bool>(),
           py::arg("strength"), py::arg("blocks_start"),
           py::arg("blocks_length"), py::arg("start"), py::arg("end"),
           py::arg("positive"))
      .def("compare",
           [](ProxType &self, ProxType &other) {
             return static_cast<bool>(self.compare(other));
           })
      .def("__eq__",
           [](ProxType &self, ProxType &other) {
             return static_cast<bool>(self.compare(other));
           });
}

template <typename Scalar>
auto bind_prox_hierarchy(py::module_ &m, const char *prox_name,
                         const char *prox_zero_name,
                         const char *prox_positive_name,
                         const char *prox_l1_name, const char *prox_l2_name,
                         const char *prox_l2sq_name,
                         const char *prox_elasticnet_name,
                         const char *prox_tv_name,
                         const char *prox_equality_name,
                         const char *prox_l1w_name,
                         const char *prox_group_l1_name,
                         const char *prox_binarsity_name,
                         const char *prox_multi_name,
                         const char *prox_sorted_l1_name,
                         const char *prox_slope_name) {
  using ProxBase = TProx<Scalar, Scalar>;
  using ProxZero = TProxZero<Scalar, Scalar>;
  using ProxPositive = TProxPositive<Scalar, Scalar>;
  using ProxL1 = TProxL1<Scalar, Scalar>;
  using ProxL2 = TProxL2<Scalar, Scalar>;
  using ProxL2Sq = TProxL2Sq<Scalar, Scalar>;
  using ProxElasticNet = TProxElasticNet<Scalar, Scalar>;
  using ProxTV = TProxTV<Scalar, Scalar>;
  using ProxEquality = TProxEquality<Scalar, Scalar>;
  using ProxL1w = TProxL1w<Scalar, Scalar>;
  using ProxWithGroups = TProxWithGroups<Scalar, Scalar>;
  using ProxGroupL1 = TProxGroupL1<Scalar, Scalar>;
  using ProxBinarsity = TProxBinarsity<Scalar, Scalar>;
  using ProxMulti = TProxMulti<Scalar, Scalar>;
  using ProxSortedL1 = TProxSortedL1<Scalar, Scalar>;
  using ProxSlope = TProxSlope<Scalar, Scalar>;

  bind_prox_base<ProxBase, Scalar>(m, prox_name);

  py::class_<ProxWithGroups, py::smart_holder, ProxBase>(
      m, std::string(prox_name).append("WithGroups").c_str())
      .def("set_blocks_start", &ProxWithGroups::set_blocks_start,
           py::arg("blocks_start"))
      .def("set_blocks_length", &ProxWithGroups::set_blocks_length,
           py::arg("blocks_length"));

  auto prox_zero =
      py::class_<ProxZero, py::smart_holder, ProxBase>(m,
                                                                prox_zero_name)
      .def(py::init<Scalar>(), py::arg("strength"))
      .def(py::init<Scalar, ulong, ulong>(), py::arg("strength"),
           py::arg("start"), py::arg("end"))
      .def("compare",
           [](ProxZero &self, ProxZero &other) {
             return static_cast<bool>(self.compare(other));
           })
      .def("__eq__",
           [](ProxZero &self, ProxZero &other) {
             return static_cast<bool>(self.compare(other));
           });
  tick::pybind::enable_cereal_pickle<ProxZero>(prox_zero);

  auto prox_positive =
      py::class_<ProxPositive, py::smart_holder, ProxBase>(
          m, prox_positive_name)
      .def(py::init<Scalar>(), py::arg("strength"))
      .def(py::init<Scalar, ulong, ulong>(), py::arg("strength"),
           py::arg("start"), py::arg("end"))
      .def("compare",
           [](ProxPositive &self, ProxPositive &other) {
             return static_cast<bool>(self.compare(other));
           })
      .def("__eq__",
           [](ProxPositive &self, ProxPositive &other) {
             return static_cast<bool>(self.compare(other));
           });
  tick::pybind::enable_cereal_pickle<ProxPositive>(prox_positive);

  auto prox_l1 =
      bind_basic_prox<ProxL1, ProxBase, Scalar>(m, prox_l1_name);
  auto prox_l2 =
      bind_basic_prox<ProxL2, ProxBase, Scalar>(m, prox_l2_name);
  auto prox_l2sq =
      bind_basic_prox<ProxL2Sq, ProxBase, Scalar>(m, prox_l2sq_name);
  auto prox_tv =
      bind_basic_prox<ProxTV, ProxBase, Scalar>(m, prox_tv_name);
  auto prox_equality =
      bind_basic_prox<ProxEquality, ProxBase, Scalar>(m, prox_equality_name);
  bind_array_step_call<decltype(prox_l1), ProxL1, Scalar>(prox_l1);
  bind_array_step_call<decltype(prox_l2sq), ProxL2Sq, Scalar>(prox_l2sq);
  tick::pybind::enable_cereal_pickle<ProxL1>(prox_l1);
  tick::pybind::enable_cereal_pickle<ProxL2>(prox_l2);
  tick::pybind::enable_cereal_pickle<ProxL2Sq>(prox_l2sq);
  tick::pybind::enable_cereal_pickle<ProxTV>(prox_tv);
  tick::pybind::enable_cereal_pickle<ProxEquality>(prox_equality);

  auto prox_elasticnet =
      py::class_<ProxElasticNet, py::smart_holder, ProxBase>(
          m, prox_elasticnet_name)
      .def(py::init<Scalar, Scalar, bool>(), py::arg("strength"),
           py::arg("ratio"), py::arg("positive"))
      .def(py::init<Scalar, Scalar, ulong, ulong, bool>(),
           py::arg("strength"), py::arg("ratio"), py::arg("start"),
           py::arg("end"), py::arg("positive"))
      .def("get_ratio", &ProxElasticNet::get_ratio)
      .def("set_ratio", &ProxElasticNet::set_ratio, py::arg("ratio"))
      .def("compare",
           [](ProxElasticNet &self, ProxElasticNet &other) {
             return static_cast<bool>(self.compare(other));
           })
      .def("__eq__",
           [](ProxElasticNet &self, ProxElasticNet &other) {
             return static_cast<bool>(self.compare(other));
           });
  tick::pybind::enable_cereal_pickle<ProxElasticNet>(prox_elasticnet);

  auto prox_l1w =
      py::class_<ProxL1w, py::smart_holder, ProxBase>(
          m, prox_l1w_name, py::dynamic_attr())
      .def(py::init<Scalar, std::shared_ptr<SArray<Scalar>>, bool>(),
           py::arg("strength"), py::arg("weights"), py::arg("positive"))
      .def(py::init<Scalar, std::shared_ptr<SArray<Scalar>>, ulong, ulong,
                    bool>(),
           py::arg("strength"), py::arg("weights"), py::arg("start"),
           py::arg("end"), py::arg("positive"))
      .def("set_weights", &ProxL1w::set_weights, py::arg("weights"))
      .def("compare",
           [](ProxL1w &self, ProxL1w &other) {
             return static_cast<bool>(self.compare(other));
           })
      .def("__eq__",
           [](ProxL1w &self, ProxL1w &other) {
             return static_cast<bool>(self.compare(other));
           });
  bind_array_step_call<decltype(prox_l1w), ProxL1w, Scalar>(prox_l1w);
  tick::pybind::enable_cereal_pickle<ProxL1w>(prox_l1w);

  auto prox_group_l1 =
      py::class_<ProxGroupL1, py::smart_holder, ProxWithGroups>(
          m, prox_group_l1_name)
          .def(py::init<Scalar, SArrayULongPtr, SArrayULongPtr, bool>(),
               py::arg("strength"), py::arg("blocks_start"),
               py::arg("blocks_length"), py::arg("positive"))
          .def(py::init<Scalar, SArrayULongPtr, SArrayULongPtr, ulong, ulong,
                        bool>(),
               py::arg("strength"), py::arg("blocks_start"),
               py::arg("blocks_length"), py::arg("start"), py::arg("end"),
               py::arg("positive"))
          .def("compare",
               [](ProxGroupL1 &self, ProxGroupL1 &other) {
                 return static_cast<bool>(self.compare(other));
               })
          .def("__eq__",
               [](ProxGroupL1 &self, ProxGroupL1 &other) {
                 return static_cast<bool>(self.compare(other));
               });
  tick::pybind::enable_cereal_pickle<ProxGroupL1>(prox_group_l1);

  auto prox_binarsity =
      py::class_<ProxBinarsity, py::smart_holder, ProxWithGroups>(
          m, prox_binarsity_name)
          .def(py::init<Scalar, SArrayULongPtr, SArrayULongPtr, bool>(),
               py::arg("strength"), py::arg("blocks_start"),
               py::arg("blocks_length"), py::arg("positive"))
          .def(py::init<Scalar, SArrayULongPtr, SArrayULongPtr, ulong, ulong,
                        bool>(),
               py::arg("strength"), py::arg("blocks_start"),
               py::arg("blocks_length"), py::arg("start"), py::arg("end"),
               py::arg("positive"))
          .def("compare",
               [](ProxBinarsity &self, ProxBinarsity &other) {
                 return static_cast<bool>(self.compare(other));
               })
          .def("__eq__",
               [](ProxBinarsity &self, ProxBinarsity &other) {
                 return static_cast<bool>(self.compare(other));
               });
  tick::pybind::enable_cereal_pickle<ProxBinarsity>(prox_binarsity);

  auto prox_multi = py::class_<ProxMulti, py::smart_holder, ProxBase>(
                        m, prox_multi_name)
      .def(py::init<std::vector<std::shared_ptr<ProxBase>>>(), py::arg("proxs"))
      .def("compare",
           [](ProxMulti &self, ProxMulti &other) {
             return static_cast<bool>(self.compare(other));
           })
      .def("__eq__",
           [](ProxMulti &self, ProxMulti &other) {
             return static_cast<bool>(self.compare(other));
           });
  tick::pybind::enable_cereal_pickle<ProxMulti>(prox_multi);

  auto prox_sorted_l1 =
      py::class_<ProxSortedL1, py::smart_holder, ProxBase>(
          m, prox_sorted_l1_name)
      .def(py::init<Scalar, WeightsType, bool>(), py::arg("strength"),
           py::arg("weights_type"), py::arg("positive"))
      .def(py::init<Scalar, WeightsType, ulong, ulong, bool>(),
           py::arg("strength"), py::arg("weights_type"), py::arg("start"),
           py::arg("end"), py::arg("positive"))
      .def("get_weights_type", &ProxSortedL1::get_weights_type)
      .def("set_weights_type", &ProxSortedL1::set_weights_type,
           py::arg("weights_type"))
      .def("get_weight_i", &ProxSortedL1::get_weight_i, py::arg("i"))
      .def("compare",
           [](ProxSortedL1 &self, ProxSortedL1 &other) {
             return static_cast<bool>(self.compare(other));
           })
      .def("__eq__",
           [](ProxSortedL1 &self, ProxSortedL1 &other) {
             return static_cast<bool>(self.compare(other));
           });
  tick::pybind::enable_cereal_pickle<ProxSortedL1>(prox_sorted_l1);

  auto prox_slope =
      py::class_<ProxSlope, py::smart_holder, ProxSortedL1>(
          m, prox_slope_name)
      .def(py::init<Scalar, Scalar, bool>(), py::arg("strength"),
           py::arg("false_discovery_rate"), py::arg("positive"))
      .def(py::init<Scalar, Scalar, ulong, ulong, bool>(),
           py::arg("strength"), py::arg("false_discovery_rate"),
           py::arg("start"), py::arg("end"), py::arg("positive"))
      .def("get_false_discovery_rate", &ProxSlope::get_false_discovery_rate)
      .def("set_false_discovery_rate", &ProxSlope::set_false_discovery_rate,
           py::arg("false_discovery_rate"))
      .def("compare",
           [](ProxSlope &self, ProxSlope &other) {
             return static_cast<bool>(self.compare(other));
           })
      .def("__eq__",
           [](ProxSlope &self, ProxSlope &other) {
             return static_cast<bool>(self.compare(other));
           });
  tick::pybind::enable_cereal_pickle<ProxSlope>(prox_slope);
}

}  // namespace

PYBIND11_MODULE(prox, m) {
  tick::pybind::ensure_numpy_imported();

  py::enum_<WeightsType>(m, "WeightsType")
      .value("bh", WeightsType::bh)
      .value("oscar", WeightsType::oscar);
  m.attr("WeightsType_bh") = py::cast(WeightsType::bh);
  m.attr("WeightsType_oscar") = py::cast(WeightsType::oscar);

  bind_prox_hierarchy<double>(
      m, "ProxDouble", "ProxZeroDouble", "ProxPositiveDouble",
      "ProxL1Double", "ProxL2Double", "ProxL2SqDouble",
      "ProxElasticNetDouble", "ProxTVDouble", "ProxEqualityDouble",
      "ProxL1wDouble", "ProxGroupL1Double", "ProxBinarsityDouble",
      "ProxMultiDouble", "ProxSortedL1Double", "ProxSlopeDouble");

  bind_prox_hierarchy<float>(
      m, "ProxFloat", "ProxZeroFloat", "ProxPositiveFloat", "ProxL1Float",
      "ProxL2Float", "ProxL2SqFloat", "ProxElasticNetFloat", "ProxTVFloat",
      "ProxEqualityFloat", "ProxL1wFloat", "ProxGroupL1Float",
      "ProxBinarsityFloat", "ProxMultiFloat", "ProxSortedL1Float",
      "ProxSlopeFloat");
}
