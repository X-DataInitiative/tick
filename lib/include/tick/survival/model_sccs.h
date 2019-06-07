//
// Created by Maryan Morel on 18/05/2017.
//

#ifndef LIB_INCLUDE_TICK_SURVIVAL_MODEL_SCCS_H_
#define LIB_INCLUDE_TICK_SURVIVAL_MODEL_SCCS_H_

// License: BSD 3 clause

#include <fstream>

#include "tick/base/base.h"
#include "tick/array/sarray2d.h"
#include "tick/array/ssparsearray2d.h"
#include "tick/base_model/model_lipschitz.h"

class DLL_PUBLIC ModelSCCS : public ModelLipschitz {
 public:
  using ModelLipschitz::get_class_name;
  friend class cereal::access;

 protected:
  ulong n_intervals;
  SArrayULongPtr n_lags;
  ArrayULong col_offset;
  ulong n_samples;
  ulong n_observations;
  ulong n_lagged_features;
  ulong n_features;

  // Label vectors
  SArrayIntPtrList1D labels;

  // Feature matrices
  SBaseArrayDouble2dPtrList1D features;

  // Censoring vectors
  SArrayULongPtr censoring;

 public:
  ModelSCCS() {}  // for cereal
  ModelSCCS(const SBaseArrayDouble2dPtrList1D &features, const SArrayIntPtrList1D &labels,
            const SArrayULongPtr censoring, const SArrayULongPtr n_lags);

  double loss(const ArrayDouble &coeffs) override;

  double loss_i(const ulong i, const ArrayDouble &coeffs) override;

  void grad(const ArrayDouble &coeffs, ArrayDouble &out) override;

  void grad_i(const ulong i, const ArrayDouble &coeffs,
              ArrayDouble &out) override;

  void compute_lip_consts() override;

  ulong get_n_samples() const override { return n_samples; }

  ulong get_n_features() const override { return n_features; }

  ulong get_rand_max() { return n_samples; }

  ulong get_epoch_size() const override { return n_samples; }

  // Number of parameters to be estimated. Can differ from the number of
  // features, e.g. when using lags.
  ulong get_n_coeffs() const override { return n_lagged_features; }

  inline ulong get_max_interval(ulong i) const {
    return std::min(censoring->value(i), n_intervals);
  }

  bool is_sparse() const override { return false; }

  inline BaseArrayDouble get_longitudinal_features(ulong i, ulong t) const {
    return view_row(*features[i], t);
  }

  inline double get_longitudinal_label(ulong i, ulong t) const {
    return view(*labels[i])[t];
  }

  double get_inner_prod(const ulong i, const ulong t,
                        const ArrayDouble &coeffs) const;

  static inline double sumExpMinusMax(ArrayDouble &x, double x_max) {
    double sum = 0;
    for (ulong i = 0; i < x.size(); ++i)
      sum += exp(x[i] - x_max);  // overflow-proof
    return sum;
  }

  static inline double logSumExp(ArrayDouble &x) {
    double x_max = x.max();
    return x_max + log(sumExpMinusMax(x, x_max));
  }

  static inline void softMax(ArrayDouble &x, ArrayDouble &out) {
    double x_max = x.max();
    double sum = sumExpMinusMax(x, x_max);
    for (ulong i = 0; i < x.size(); i++) {
      out[i] = exp(x[i] - x_max) / sum;  // overflow-proof
    }
  }

  template <class Archive>
  void load(Archive &ar) {
    ar(cereal::make_nvp("ModelSCCS",
                        typename cereal::base_class<TModelLipschitz<double, double>>(this)));
    bool sparse_features = false;
    ar(sparse_features);
    ar(n_samples, n_features, n_observations, n_lagged_features, n_intervals);
    if (!n_lags) n_lags = std::make_shared<SArrayULong>();
    if (!censoring) censoring = std::make_shared<SArrayULong>();
    ar(*n_lags, col_offset);
    labels.clear();
    for (size_t i = 0; i < n_samples; i++) {
      labels.emplace_back(std::make_shared<SArrayInt>());
      labels.back()->load(ar);
    }
    features.clear();
    for (size_t i = 0; i < n_samples; i++) {
      if(sparse_features) {
        auto ptr = std::make_shared<SSparseArray2d<double>>();
        ptr->load(ar);
        features.emplace_back(std::move(ptr));
      }else {
        auto ptr = std::make_shared<      SArray2d<double>>();
        ptr->load(ar);
        features.emplace_back(std::move(ptr));
      }
    }
    ar(cereal::make_nvp("censoring", *censoring));
  }

  template <class Archive>
  void save(Archive &ar) const {
    ar(cereal::make_nvp("ModelSCCS",
                        typename cereal::base_class<TModelLipschitz<double, double>>(this)));
    bool sparse_features = features[0]->is_sparse();
    ar(sparse_features);
    KLOG(INF) << sparse_features;
    ar(n_samples, n_features, n_observations, n_lagged_features, n_intervals);
    ar(*n_lags, col_offset);
    for (size_t i = 0; i < n_samples; i++) labels[i]->save(ar);
    for (size_t i = 0; i < n_samples; i++) {
      if(sparse_features) static_cast<SSparseArray2d<double>*>(features[i].get())->save(ar);
      else static_cast<SArray2d<double>*>(features[i].get())->save(ar);
    }
    ar(cereal::make_nvp("censoring", *censoring));
  }

  void save_as_pb(std::string _file) {
    if (!n_lags) n_lags = std::make_shared<SArrayULong>();
    if (!censoring) censoring = std::make_shared<SArrayULong>();
    std::ofstream ss(_file, std::ios::out | std::ios::binary);
    cereal::PortableBinaryOutputArchive ar(ss);
    save(ar);
  }

  BoolStrReport compare(const ModelSCCS &that, std::stringstream &ss) {
    ss << get_class_name() << std::endl;
    auto are_equal = ModelLipschitz::compare(that, ss) && TICK_CMP_REPORT(ss, n_samples) &&
                     TICK_CMP_REPORT(ss, n_features) && TICK_CMP_REPORT(ss, n_observations) &&
                     TICK_CMP_REPORT(ss, n_lagged_features) && TICK_CMP_REPORT(ss, n_intervals) &&
                     TICK_CMP_REPORT_PTR(ss, n_lags) && TICK_CMP_REPORT(ss, col_offset) &&
                     TICK_CMP_REPORT_VECTOR_SPTR_1D(ss, labels, int32_t) &&
                     TICK_CMP_REPORT_VECTOR_ARRAY2D(ss, features, double) &&
                     TICK_CMP_REPORT_PTR(ss, censoring);
    return BoolStrReport(are_equal, ss.str());
  }
  BoolStrReport compare(const ModelSCCS &that) {
    std::stringstream ss;
    return compare(that, ss);
  }
  BoolStrReport operator==(const ModelSCCS &that) { return ModelSCCS::compare(that); }
};

CEREAL_SPECIALIZE_FOR_ALL_ARCHIVES(ModelSCCS, cereal::specialization::member_load_save)
CEREAL_REGISTER_TYPE(ModelSCCS)

#endif  // LIB_INCLUDE_TICK_SURVIVAL_MODEL_SCCS_H_
