//
// Created by Stéphane GAIFFAS on 17/03/2016.
//

#ifndef LIB_INCLUDE_TICK_BASE_MODEL_MODEL_LIPSCHITZ_H_
#define LIB_INCLUDE_TICK_BASE_MODEL_MODEL_LIPSCHITZ_H_

// License: BSD 3 clause

#include "model.h"

/**
 * \class ModelLipschitz
 * \brief An interface for a Model with the ability to compute Lipschitz
 * constants
 */
template <class T, class K = T>
class DLL_PUBLIC TModelLipschitz : public virtual TModel<T, K> {
 protected:
  using TModel<T, K>::compute_lip_consts;
  using TModel<T, K>::get_class_name;

 protected:
  //! True if all lipschitz constants are already computed
  bool ready_lip_consts = false;

  //! True if the maximum of lipschitz constants is already computed
  bool ready_lip_max = false;

  //! True if the mean of lipschitz constants is already computed
  bool ready_lip_mean = false;

  //! Average and maximum Lipschitz constants
  T lip_mean = 0, lip_max = 0;

  //! All Lipschitz constants
  Array<T> lip_consts;

 public:
  TModelLipschitz();
  virtual ~TModelLipschitz() {}

  /**
   * @brief Get the maximum of all Lipschits constants
   * @note This will cache the obtained value for later calls
   */
  T get_lip_max() override;

  /**
   * @brief Get the mean of all Lipschits constants
   * @note This will cache the obtained value for later calls
   */
  T get_lip_mean() override;

  template <class Archive>
  void serialize(Archive &ar) {
    ar(CEREAL_NVP(ready_lip_consts), CEREAL_NVP(ready_lip_max),
       CEREAL_NVP(ready_lip_mean), CEREAL_NVP(lip_consts), CEREAL_NVP(lip_mean),
       CEREAL_NVP(lip_max));
  }

 protected:
  BoolStrReport compare(const TModelLipschitz<T, K> &that,
                        std::stringstream &ss) {
    return BoolStrReport(TICK_CMP_REPORT(ss, ready_lip_consts) &&
                             TICK_CMP_REPORT(ss, ready_lip_max) &&
                             TICK_CMP_REPORT(ss, ready_lip_mean) &&
                             TICK_CMP_REPORT(ss, lip_mean) &&
                             TICK_CMP_REPORT(ss, lip_max) &&
                             TICK_CMP_REPORT(ss, lip_consts),
                         ss.str());
  }
};

using ModelLipschitz = TModelLipschitz<double, double>;

using ModelLipschitzDouble = TModelLipschitz<double, double>;
using ModelLipschitzFloat = TModelLipschitz<float, float>;

#endif  // LIB_INCLUDE_TICK_BASE_MODEL_MODEL_LIPSCHITZ_H_
