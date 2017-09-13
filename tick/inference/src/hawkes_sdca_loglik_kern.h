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
  //! Either decay or decays is defined but not both
  double decay;

  //! @brief decays shared by all Hawkes sum exponential kernels
  //! Either decay or decays is defined but not both
  ArrayDouble decays;

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

  HawkesSDCALoglikKern(const ArrayDouble &decays, double l_l2sq,
                       int max_n_threads = 1, double tol = 0.,
                       RandType rand_type = RandType::unif, int seed = -1);

  //! @brief allocate buffer arrays once data has been given
  void compute_weights();

  //! @brief Perform one iteration of the algorithm
  void solve();

  bool use_sumexp_kernel() const;

  double get_decay() const;
  void set_decay(const double decay);

  SArrayDoublePtr get_decays() const;
  void set_decays(const ArrayDouble &decays);

  SArrayDoublePtr get_iterate();

  double loss(const ArrayDouble &coeffs) override;
  double current_dual_objective();

  ulong get_n_samples() const override {
    return n_jumps_per_node->sum();
  }

  ulong get_n_coeffs() const override {
    if (use_sumexp_kernel()) {
      return n_nodes + n_nodes * n_nodes * decays.size();
    } else {
      return n_nodes + n_nodes * n_nodes;
    }
  }

  ulong get_n_decays() const {
    return decays.size();
  }

 private:

  void allocate_weights();
  void compute_weights_dim_i(ulong i_r, std::shared_ptr<ArrayDouble2dList1D> G_buffer);
  void compute_weights_dim_i2(ulong i_r, std::shared_ptr<ArrayDouble2dList1D> G_buffer);
  void solve_dim_i(const ulong i);

  double loss_dim_i(const ulong i, const ArrayDouble &coeffs) const;
  double current_dual_objective_dim_i(const ulong i) const;
  void synchronize_sdca();

  ulong get_n_coeffs_per_node() const {
    if (use_sumexp_kernel()) {
      return 1 +  n_nodes * decays.size();
    } else {
      return 1 + n_nodes;
    }
  }
};

#endif  // TICK_INFERENCE_SRC_HAWKES_SDCA_LOGLIK_KERN_H_
