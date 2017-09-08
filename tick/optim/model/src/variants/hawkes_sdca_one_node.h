#ifndef TICK_HAWKES_SDCA_ONE_NODE_H
#define TICK_HAWKES_SDCA_ONE_NODE_H

// License: BSD 3 clause

#include "model.h"

class ModelHawkesSDCAOneNode : public Model {

  BaseArrayDouble2d features;
  ArrayDouble n_times_psi;

 public:
  explicit ModelHawkesSDCAOneNode(ArrayDouble2d &g_i, ArrayDouble &G_i) {
    if (g_i.n_cols() != G_i.size()) {
      TICK_ERROR("g_i and G_i must have the same number of columns");
    }

    this->features = view(g_i);
    this->n_times_psi = view(G_i);
  }

  BaseArrayDouble get_features(const ulong i) const override {
    return view_row(features, i);
  }

  ulong get_n_features() const override {
    return features.n_cols();
  }

  ulong get_n_samples() const override {
    return features.n_rows();
  }

  void sdca_primal_dual_relation(const double l_l2sq,
                                 const ArrayDouble &dual_vector,
                                 ArrayDouble &out_primal_vector) override {
    const double _1_over_lbda_n = 1 / (l_l2sq * get_n_samples());
    out_primal_vector.init_to_zero();

    for (ulong i = 0; i < get_n_samples(); ++i) {
      const BaseArrayDouble feature_i = get_features(i);
      out_primal_vector.mult_incr(feature_i, dual_vector[i] * _1_over_lbda_n);
    }
    out_primal_vector.mult_incr(n_times_psi, -_1_over_lbda_n);
  }

  double sdca_dual_min_i(const ulong i,
                         const double dual_i,
                         const ArrayDouble &primal_vector,
                         const double previous_delta_dual_i,
                         double l_l2sq) override {
    BaseArrayDouble feature_i = get_features(i);

    double normalized_features_norm = feature_i.norm_sq() / (l_l2sq * get_n_features());
    const double primal_dot_features = primal_vector.dot(feature_i);

    const double tmp = dual_i * normalized_features_norm - primal_dot_features;
    double new_dual = (std::sqrt(tmp * tmp + 4 * normalized_features_norm) + tmp);
    new_dual /= 2 * normalized_features_norm;

    return new_dual - dual_i;
  }

  ulong get_n_coeffs() const override {
    return features.n_cols();
  }
};

#endif //TICK_HAWKES_SDCA_ONE_NODE_H
