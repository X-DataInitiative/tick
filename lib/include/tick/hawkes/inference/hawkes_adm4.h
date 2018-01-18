
#ifndef LIB_INCLUDE_TICK_HAWKES_INFERENCE_HAWKES_ADM4_H_
#define LIB_INCLUDE_TICK_HAWKES_INFERENCE_HAWKES_ADM4_H_

// License: BSD 3 clause

#include "tick/base/base.h"
#include "tick/hawkes/model/base/model_hawkes_list.h"

/**
 * \class HawkesADM4
 * \brief Class for implementation of the algorithm described in the paper
 * `Learning Social Infectivity in Sparse Low-rank Networks Using
 * Multi-dimensional Hawkes Processes` by Zhou, Zha and Song (2013) in AISTATS
 */
class HawkesADM4 : public ModelHawkesList {
  //! @brief Decay shared by all Hawkes exponential kernels
  double decay;

  //! @brief Penalty parameter
  double rho;

  //! @brief Weights storing kernel integrals equal
  //! for node u : kernel_integral[u] = \sum_r \int_0^{T - t_i^u} g(s) ds
  ArrayDouble kernel_integral;

  //! @brief Weights storing sum of kernel values at specific points
  //! for realization r: g[r][u][i][v] = \sum_{t_j^v < t_i^u} g(t_i^u - t_j^v)
  ArrayDouble2dList2D g;

  //! @brief Buffer variables used to compute p_ij
  ArrayDouble2d next_C, unnormalized_next_C;

  //! @brief Buffer variables to compute next baseline (mu)
  ArrayDouble2d next_mu;

 public:
  HawkesADM4(const double decay, const double rho, const int max_n_threads = 1,
             const unsigned int optimization_level = 0);

  //! @brief allocate buffer arrays once data has been given
  void compute_weights();

  //! @brief Perform one iteration of the algorithm
  void solve(ArrayDouble &mu, ArrayDouble2d &adjacency, ArrayDouble2d &z1, ArrayDouble2d &z2,
             ArrayDouble2d &u1, ArrayDouble2d &u2);

 private:
  void compute_weights_ru(const ulong r_u, ArrayDouble2d &map_kernel_integral);

  void update_u(const ulong u, ArrayDouble &mu, ArrayDouble2d &adjacency, ArrayDouble2d &z1,
                ArrayDouble2d &z2, ArrayDouble2d &u1, ArrayDouble2d &u2);

  void estimate_ru(const ulong r_u,
                   ArrayDouble &mu, ArrayDouble2d &adjacency);

  void update_adjacency_u(const ulong u, ArrayDouble &adjacency_u,
                          ArrayDouble &z1_u, ArrayDouble &z2_u,
                          ArrayDouble &u1_u, ArrayDouble &u2_u);

  void update_baseline_u(const ulong u, ArrayDouble &mu);

 public:
  double get_decay() const;
  void set_decay(const double decay);
  double get_rho() const;
  void set_rho(const double rho);
};

#endif  // LIB_INCLUDE_TICK_HAWKES_INFERENCE_HAWKES_ADM4_H_
