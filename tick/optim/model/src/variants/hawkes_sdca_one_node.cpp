// License: BSD 3 clause

#include <math.h>
#include "hawkes_sdca_one_node.h"

ModelHawkesSDCAOneNode::ModelHawkesSDCAOneNode(ArrayDouble2d &g_i, ArrayDouble &G_i,
                                               ulong n_samples):
  n_samples(n_samples), max_dual(std::numeric_limits<double>::infinity()) {
  if (g_i.n_cols() != G_i.size()) {
    TICK_ERROR("g_i and G_i must have the same number of columns");
  }

  this->features = view(g_i);
  this->n_times_psi = view(G_i);
  features_norm_sq = ArrayDouble(features.n_rows());

  for (ulong i = 0; i < features_norm_sq.size(); ++i) {
    features_norm_sq[i] = get_features(i).norm_sq();
  }
}

BaseArrayDouble ModelHawkesSDCAOneNode::get_features(const ulong i) const {
  return view_row(features, i);
}

ulong ModelHawkesSDCAOneNode::get_n_features() const {
  return features.n_cols();
}

ulong ModelHawkesSDCAOneNode::get_n_samples() const {
  return n_samples;
}

void ModelHawkesSDCAOneNode::sdca_primal_dual_relation(const double l_l2sq,
                                                       const ArrayDouble &dual_vector,
                                                       ArrayDouble &out_primal_vector) {
  const double _1_over_lbda_n = 1 / (l_l2sq * features.n_rows());
  out_primal_vector.init_to_zero();

  for (ulong i = 0; i < features.n_rows(); ++i) {
    const BaseArrayDouble feature_i = get_features(i);
    out_primal_vector.mult_incr(feature_i, dual_vector[i] * _1_over_lbda_n);
  }
  out_primal_vector.mult_incr(n_times_psi, -_1_over_lbda_n);
}

double ModelHawkesSDCAOneNode::loss(const ArrayDouble &coeffs) {
  const ulong n_samples = get_n_samples();
  double loss = n_times_psi.dot(coeffs) / n_samples;

  for (ulong i = 0; i < features.n_rows(); ++i) {
    const double inner_product = get_features(i).dot(coeffs);
    // If one dot product is negative, returned loss is infinite
    if (inner_product <= 0) {
      return std::numeric_limits<double>::infinity();
    }
    loss -= log(get_features(i).dot(coeffs)) / n_samples;
  }
  return loss;
}

double ModelHawkesSDCAOneNode::dual_objective(const double l_l2sq, const ArrayDouble &dual) {
  const ulong n_samples = get_n_samples();

  if (dual.size() != features.n_rows()){
    TICK_ERROR("Dual vector (" << dual.size() << ") must be as long as features(" << n_samples << ")")
  }

  double dual_loss = 0;
  const double _1_over_lbda_n = 1 / (l_l2sq * n_samples);

  ArrayDouble buffer(get_n_coeffs());
  buffer.init_to_zero();
  for (ulong i = 0; i < features.n_rows(); ++i) {
    dual_loss += (1 + log(dual[i])) / n_samples;
    buffer.mult_incr(get_features(i), dual[i] * _1_over_lbda_n);
  }

  buffer.mult_incr(n_times_psi, -_1_over_lbda_n);

  dual_loss -= 0.5 * l_l2sq * buffer.norm_sq();
  return dual_loss;
}

double ModelHawkesSDCAOneNode::sdca_dual_min_i(const ulong i,
                                               const double dual_i,
                                               const ArrayDouble &primal_vector,
                                               const double previous_delta_dual_i,
                                               double l_l2sq) {
  BaseArrayDouble feature_i = get_features(i);

  double normalized_features_norm = features_norm_sq[i] / (l_l2sq * get_n_features());
  const double primal_dot_features = primal_vector.dot(feature_i);

  const double tmp = dual_i * normalized_features_norm - primal_dot_features;
  double new_dual = (std::sqrt(tmp * tmp + 4 * normalized_features_norm) + tmp);
  new_dual /= 2 * normalized_features_norm;

  if (new_dual > max_dual) {
    new_dual = max_dual;
  }

  return new_dual - dual_i;
}

ulong ModelHawkesSDCAOneNode::get_n_coeffs() const {
  return features.n_cols();
}

void ModelHawkesSDCAOneNode::set_max_dual(const double max_dual){
  this->max_dual = max_dual;
}