//
// Created by Martin Bompaire on 21/10/15.
//

#ifndef LIB_INCLUDE_TICK_LINEAR_MODEL_MODEL_POISREG_H_
#define LIB_INCLUDE_TICK_LINEAR_MODEL_MODEL_POISREG_H_

// License: BSD 3 clause

#include "tick/base_model/model_generalized_linear.h"

// TODO: labels should be a ArrayUInt

enum class LinkType {
  identity = 0,
  exponential
};

class DLL_PUBLIC ModelPoisReg : public ModelGeneralizedLinear {
 private:
  LinkType link_type;
  bool ready_non_zero_label_map;
  VArrayULongPtr non_zero_labels;
  ulong n_non_zeros_labels;

 public:
  ModelPoisReg(const SBaseArrayDouble2dPtr features,
               const SArrayDoublePtr labels,
               const LinkType link_type,
               const bool fit_intercept,
               const int n_threads = 1);

  const char *get_class_name() const override {
    return "ModelPoisReg";
  }

  double loss_i(const ulong i, const ArrayDouble &coeffs) override;

  double grad_i_factor(const ulong i, const ArrayDouble &coeffs) override;

  double sdca_dual_min_i(const ulong i,
                         const double dual_i,
                         const ArrayDouble &primal_vector,
                         const double previous_delta_dual_i,
                         double l_l2sq) override;

  void sdca_primal_dual_relation(const double l_l2sq,
                                 const ArrayDouble &dual_vector,
                                 ArrayDouble &out_primal_vector) override;

  /**
   * Returns a mapping from the sampled observation (in [0, rand_max)) to the observation
   * position (in [0, n_samples)). For identity link this is needed as zero labeled observations
   * are discarded in SDCA.
   * For exponential link nullptr is returned, it means no index_map is required as the mapping
   * is the canonical inedx_map[i] = i
   */
  SArrayULongPtr get_sdca_index_map() override {
    if (link_type == LinkType::exponential) {
      return nullptr;
    }
    if (!ready_non_zero_label_map) init_non_zero_label_map();
    return non_zero_labels;
  }

 private:
  double sdca_dual_min_i_exponential(const ulong i,
                                     const double dual_i,
                                     const ArrayDouble &primal_vector,
                                     const double previous_delta_dual_i,
                                     double l_l2sq);
  /**
   * @brief Initialize the hash map that allow fast retrieving for get_non_zero_i
   */
  void init_non_zero_label_map();

  double sdca_dual_min_i_identity(const ulong i,
                                  const double dual_i,
                                  const ArrayDouble &primal_vector,
                                  const double previous_delta_dual_i,
                                  double l_l2sq);

 public:
  virtual void set_link_type(const LinkType link_type) {
    this->link_type = link_type;
  }

  virtual LinkType get_link_type() {
    return link_type;
  }
};

#endif  // LIB_INCLUDE_TICK_LINEAR_MODEL_MODEL_POISREG_H_
