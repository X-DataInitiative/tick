//
// Created by Stéphane GAIFFAS on 12/04/2016.
//

#ifndef LIB_INCLUDE_TICK_SURVIVAL_MODEL_COXREG_PARTIAL_LIK_H_
#define LIB_INCLUDE_TICK_SURVIVAL_MODEL_COXREG_PARTIAL_LIK_H_

// License: BSD 3 clause

#include "tick/base_model/model.h"

template <class T, class K = T>
class DLL_PUBLIC TModelCoxRegPartialLik : public TModel<T, K> {
 public:
  friend class cereal::access;
  using TModel<T, K>::get_class_name;

 private:
  Array<T> inner_prods;
  Array<T> s1;
  ArrayULong idx;

 protected:
  ulong n_samples, n_features, n_failures;

  std::shared_ptr<BaseArray2d<T> > features;
  Array<T> times;
  ArrayUShort censoring;
  ArrayULong idx_failures;

  inline BaseArray<T> get_feature(ulong i) const {
    return view_row(*features, idx[i]);
  }

  inline T get_time(ulong i) const { return times[i]; }

  inline ushort get_censoring(ulong i) const { return censoring[i]; }

  inline ulong get_idx_failure(ulong i) const { return idx_failures[i]; }

 public:
  TModelCoxRegPartialLik() {}  // for cereal
  TModelCoxRegPartialLik(const std::shared_ptr<BaseArray2d<T> > features,
                         const std::shared_ptr<SArray<T> > times, const SArrayUShortPtr censoring);

  /**
   * \brief Computation of the value of minus the partial Cox
   * log-likelihood at
   * point coeffs.
   * It should be overflow proof and fast.
   *
   * \note
   * This code assumes that the times are inversely sorted and that the
   * rows of the features matrix and index of failure times are sorted
   * accordingly. This sorting is done automatically by the SurvData object.
   *
   * \param coeffs : The vector at which the loss is computed
   */
  T loss(const Array<K> &coeffs) override;

  void grad(const Array<K> &coeffs, Array<T> &out) override;

  template <class Archive>
  void load(Archive &ar) {
    ar(cereal::make_nvp("ModelCoxRegPartialLik", typename cereal::base_class<TModel<T, K> >(this)));
    ar(n_samples, n_features, n_failures);
    ar(inner_prods, s1, idx);
    ar(times, censoring, idx_failures);

    BaseArray2d<T> tmp_features;
    ar(cereal::make_nvp("features", tmp_features));
    features = tmp_features.as_sarray2d_ptr();
  }

  template <class Archive>
  void save(Archive &ar) const {
    ar(cereal::make_nvp("ModelCoxRegPartialLik", typename cereal::base_class<TModel<T, K> >(this)));
    ar(n_samples, n_features, n_failures);
    ar(inner_prods, s1, idx);
    ar(times, censoring, idx_failures);

    ar(cereal::make_nvp("features", *features));
  }

  BoolStrReport compare(const TModelCoxRegPartialLik<T, K> &that, std::stringstream &ss) {
    ss << get_class_name() << std::endl;
    auto are_equal = TICK_CMP_REPORT(ss, n_samples) && TICK_CMP_REPORT(ss, n_features) &&
                     TICK_CMP_REPORT(ss, n_failures) && TICK_CMP_REPORT(ss, inner_prods) &&
                     TICK_CMP_REPORT(ss, s1) && TICK_CMP_REPORT(ss, idx) &&
                     TICK_CMP_REPORT_PTR(ss, features) && TICK_CMP_REPORT(ss, times) &&
                     TICK_CMP_REPORT(ss, censoring) && TICK_CMP_REPORT(ss, idx_failures);
    return BoolStrReport(are_equal, ss.str());
  }
  BoolStrReport compare(const TModelCoxRegPartialLik<T, K> &that) {
    std::stringstream ss;
    return compare(that, ss);
  }
  BoolStrReport operator==(const TModelCoxRegPartialLik<T, K> &that) {
    return TModelCoxRegPartialLik<T, K>::compare(that);
  }
};

using ModelCoxRegPartialLik = TModelCoxRegPartialLik<double>;
using ModelCoxRegPartialLikPtr = std::shared_ptr<ModelCoxRegPartialLik>;

using ModelCoxRegPartialLikDouble = TModelCoxRegPartialLik<double, double>;
using ModelCoxRegPartialLikDoublePtr = std::shared_ptr<ModelCoxRegPartialLikDouble>;

using ModelCoxRegPartialLikFloat = TModelCoxRegPartialLik<float, float>;
using ModelCoxRegPartialLikFloatPtr = std::shared_ptr<ModelCoxRegPartialLikFloat>;

CEREAL_SPECIALIZE_FOR_ALL_ARCHIVES(ModelCoxRegPartialLikDouble,
                                   cereal::specialization::member_load_save)
CEREAL_REGISTER_TYPE(ModelCoxRegPartialLikDouble)

CEREAL_SPECIALIZE_FOR_ALL_ARCHIVES(ModelCoxRegPartialLikFloat,
                                   cereal::specialization::member_load_save)
CEREAL_REGISTER_TYPE(ModelCoxRegPartialLikFloat)

#endif  // LIB_INCLUDE_TICK_SURVIVAL_MODEL_COXREG_PARTIAL_LIK_H_
