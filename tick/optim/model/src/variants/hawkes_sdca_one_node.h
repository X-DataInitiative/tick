#ifndef TICK_HAWKES_SDCA_ONE_NODE_H
#define TICK_HAWKES_SDCA_ONE_NODE_H

// License: BSD 3 clause

#include "model.h"

class ModelHawkesSDCAOneNode : public Model {

  BaseArrayDouble2d features;
  ArrayDouble n_times_psi;
  ulong n_samples;
  ArrayDouble features_norm_sq;

 public:
  ModelHawkesSDCAOneNode(ArrayDouble2d &g_i, ArrayDouble &G_i, ulong n_samples);

  BaseArrayDouble get_features(const ulong i) const override;

  ulong get_n_features() const override;

  ulong get_n_samples() const override;

  void sdca_primal_dual_relation(const double l_l2sq,
                                 const ArrayDouble &dual_vector,
                                 ArrayDouble &out_primal_vector) override;

  double sdca_dual_min_i(const ulong i,
                         const double dual_i,
                         const ArrayDouble &primal_vector,
                         const double previous_delta_dual_i,
                         double l_l2sq) override;

  double dual_objective(const double l_l2sq, const ArrayDouble &dual) override;

  double loss(const ArrayDouble &coeffs) override;

  ulong get_n_coeffs() const override;
};

#endif //TICK_HAWKES_SDCA_ONE_NODE_H
