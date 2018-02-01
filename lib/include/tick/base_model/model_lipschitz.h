//
// Created by Stéphane GAIFFAS on 17/03/2016.
//

#ifndef LIB_INCLUDE_TICK_BASE_MODEL_MODEL_LIPSCHITZ_H_
#define LIB_INCLUDE_TICK_BASE_MODEL_MODEL_LIPSCHITZ_H_

// License: BSD 3 clause

#include "model.h"

/**
 * \class ModelLipschitz
 * \brief An interface for a Model with the ability to compute Lipschitz constants
 */
template <class T, class K = T>
class DLL_PUBLIC TModelLipschitz : public virtual TModel<T, K> {
 protected:
  using TModel<T, K>::compute_lip_consts;

 protected:
  //! True if all lipschitz constants are already computed
  bool ready_lip_consts;

  //! True if the maximum of lipschitz constants is already computed
  bool ready_lip_max;

  //! True if the mean of lipschitz constants is already computed
  bool ready_lip_mean;

  //! Average and maximum Lipschitz constants
  K lip_mean, lip_max;

  //! All Lipschitz constants
  Array<K> lip_consts;

 public:
  TModelLipschitz();
  virtual ~TModelLipschitz() {}

  const char *get_class_name() const override {
    return "ModelLipchitz";
  }

  /**
   * @brief Get the maximum of all Lipschits constants
   * @note This will cache the obtained value for later calls
   */
  K get_lip_max() override;

  /**
   * @brief Get the mean of all Lipschits constants
   * @note This will cache the obtained value for later calls
   */
  K get_lip_mean() override;

  template<class Archive>
  void serialize(Archive & ar) {
    ar(CEREAL_NVP(ready_lip_consts),
       CEREAL_NVP(ready_lip_max),
       CEREAL_NVP(ready_lip_mean),
       CEREAL_NVP(lip_consts),
       CEREAL_NVP(lip_mean),
       CEREAL_NVP(lip_max));
  }
};

class DLL_PUBLIC ModelLipschitz : public TModelLipschitz<double, double> {
 public:
  ModelLipschitz();
};

using ModelLipschitzDouble = TModelLipschitz<double, double>;
using ModelLipschitzFloat  = TModelLipschitz<float , float>;

#endif  // LIB_INCLUDE_TICK_BASE_MODEL_MODEL_LIPSCHITZ_H_
