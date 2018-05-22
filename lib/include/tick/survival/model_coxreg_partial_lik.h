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
  TModelCoxRegPartialLik(const std::shared_ptr<BaseArray2d<T> > features,
                         const std::shared_ptr<SArray<T> > times,
                         const SArrayUShortPtr censoring);

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
};

using ModelCoxRegPartialLik = TModelCoxRegPartialLik<double>;
using ModelCoxRegPartialLikPtr = std::shared_ptr<ModelCoxRegPartialLik>;

using ModelCoxRegPartialLikDouble = TModelCoxRegPartialLik<double>;
using ModelCoxRegPartialLikDoublePtr =
    std::shared_ptr<ModelCoxRegPartialLikDouble>;

using ModelCoxRegPartialLikFloat = TModelCoxRegPartialLik<double>;
using ModelCoxRegPartialLikFloatPtr =
    std::shared_ptr<ModelCoxRegPartialLikFloat>;

#endif  // LIB_INCLUDE_TICK_SURVIVAL_MODEL_COXREG_PARTIAL_LIK_H_
