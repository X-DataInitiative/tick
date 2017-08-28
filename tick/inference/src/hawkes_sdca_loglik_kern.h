#ifndef TICK_INFERENCE_SRC_HAWKES_SDCA_LOGLIK_KERN_H_
#define TICK_INFERENCE_SRC_HAWKES_SDCA_LOGLIK_KERN_H_

// License: BSD 3 clause

#include "base.h"
#include "sdca.h"
#include "base/hawkes_list.h"

/**
 * \class HawkesSDCALoglikKern
 * \brief TODO fill
 */
class HawkesSDCALoglikKern : public ModelHawkesList {
  //! @brief Decay shared by all Hawkes exponential kernels
  double decay;

  //! @brief kernel intensity of node j on node i at time t_i_k
  ArrayDouble2dList1D g;

  //! @brief compensator of kernel intensity of node j on node i between 0 and end_time
  ArrayDoubleList1D G;

  std::vector<SDCA> sdca_list;

  bool weights_allocated;

  // SDCA attributes
  double l_l2sq;
  double tol;
  RandType rand_type;
  int seed;

 public:
  HawkesSDCALoglikKern(double decay, double l_l2sq,
                       int max_n_threads = 1, double tol = 0.,
                       RandType rand_type = RandType::unif, int seed = -1);

  //! @brief allocate buffer arrays once data has been given
  void compute_weights();

  //! @brief Perform one iteration of the algorithm
  void solve(ArrayDouble &mu, ArrayDouble2d &adjacency, ArrayDouble2d &z1, ArrayDouble2d &z2,
             ArrayDouble2d &u1, ArrayDouble2d &u2);

  double get_decay() const;
  void set_decay(double decay);

 private:
  void allocate_weights();
  void compute_weights_dim_i(ulong i_r, std::shared_ptr<ArrayDouble2dList1D> G_buffer);
};

class ModelHawkesSDCAOneNode : public Model {

  BaseArrayDouble2d features;
  ArrayDouble n_times_psi;

 public:
  explicit ModelHawkesSDCAOneNode(ArrayDouble2d &g_i, ArrayDouble &G_i) {
    // TODO: would it cost less to store it in this class?
    // It would neccessit smaller chunks of contiguous data
    this->features = view(g_i);
    this->n_times_psi = view(G_i);
  }

  BaseArrayDouble get_features(const ulong i) const override {
//    return view_row(features, i);
    return BaseArrayDouble(0);
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

    ulong n_non_zero_labels_seen = 0;
    for (ulong i = 0; i < get_n_samples(); ++i) {
      const BaseArrayDouble feature_i = get_features(i);

      out_primal_vector.mult_incr(feature_i, dual_vector[i] * _1_over_lbda_n);
    }
    out_primal_vector.mult_incr(n_times_psi, - _1_over_lbda_n);
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
};

#endif  // TICK_INFERENCE_SRC_HAWKES_SDCA_LOGLIK_KERN_H_
