
#ifndef LIB_INCLUDE_TICK_HAWKES_INFERENCE_HAWKES_SUMGAUSSIANS_H_
#define LIB_INCLUDE_TICK_HAWKES_INFERENCE_HAWKES_SUMGAUSSIANS_H_

// License: BSD 3 clause

#include "tick/base/base.h"
#include "tick/hawkes/model/base/model_hawkes_list.h"

/**
 * \class HawkesSumGaussians
 * \brief Class for implementation of the algorithm described in the paper
 * `Learning Granger Causality for Hawkes Processes`
 *  by Xu, Farajtabar, and Zha (2016) in ICML
 */
class HawkesSumGaussians : public ModelHawkesList {
  //! @brief Number of gaussian functions to approximate each kernel
  ulong n_gaussians;

  //! @brief Upper bound of the last mean of gaussian functions. This is a proxy for the kernel support.
  double max_mean_gaussian;

  //! @brief Standard deviation of the gaussian functions. This bandlimit
  //! is computed using Silverman's rule of thumb.
  double std_gaussian = 1.;

  //! @brief Useful constants that appear in weights computation.
  double std_gaussian_sq = std_gaussian * std_gaussian;
  double norm_constant_gauss = std_gaussian * std::sqrt(2. * M_PI);
  double norm_constant_erf = std_gaussian * std::sqrt(2);

  //! @brief Step size used in update formulas (7) and (8)
  double step_size;

  //! @brief Number of iterations of EM steps before using proximal operators
  ulong em_max_iter;

  //! @brief Parameter controlling the Lasso (L1) regularization
  double strength_lasso;

  //! @brief Parameter controlling the Group-Lasso regularization
  double strength_grouplasso;

  //! @brief Means of the gaussian basis functions
  ArrayDouble means_gaussians;

  //! @brief Weights storing kernel integrals equal
  //! for node u : kernel_integral[u] = \sum_r \sum_m \int_0^{T - t_i^u} g_m(s) ds
  ArrayDouble kernel_integral;

  //! @brief Weights storing sum of kernel values at specific points
  //! for realization r: g[r][u][i][v*n_nodes+m] = \sum_{t_j^v < t_i^u} g_m(t_i^u - t_j^v)
  ArrayDouble2dList2D g;

  //! @brief Buffer variables used to compute p_ij
  ArrayDouble2d next_C, unnormalized_next_C;

  //! @brief Buffer variables to compute next baseline (mu)
  ArrayDouble2d next_mu;

 public:
  HawkesSumGaussians(const ulong n_gaussians, const double max_mean_gaussian,
                     const double step_size,
                     const double strength_lasso, const double strength_grouplasso,
                     const ulong em_max_iter, const int max_n_threads = 1,
                     const unsigned int optimization_level = 0);

  //! @brief allocate buffer arrays once data has been given
  void compute_weights();

  //! @brief Perform one iteration of the algorithm
  void solve(ArrayDouble &mu, ArrayDouble2d &amplitudes);

 private:
  void compute_weights_ru(const ulong r_u, ArrayDouble2d &map_kernel_integral);

  void update_u(const ulong u, ArrayDouble &mu, ArrayDouble2d &amplitudes);

  void estimate_ru(const ulong r_u,
                   ArrayDouble &mu, ArrayDouble2d &amplitudes);

  void update_amplitudes_u(const ulong u, ArrayDouble &amplitudes_u);

  void prox_amplitudes_u(const ulong u, ArrayDouble2d &amplitudes, ArrayDouble2d &amplitudes_old);

  void update_baseline_u(const ulong u, ArrayDouble &mu);

 public:
  ulong get_n_gaussians() const;
  void set_n_gaussians(const ulong n_gaussians);
  ulong get_em_max_iter() const;
  void set_em_max_iter(const ulong em_max_iter);
  double get_max_mean_gaussian() const;
  void set_max_mean_gaussian(const double max_mean_gaussian);
  double get_step_size() const;
  void set_step_size(const double step_size);
  double get_strength_lasso() const;
  void set_strength_lasso(const double strength_lasso);
  double get_strength_grouplasso() const;
  void set_strength_grouplasso(const double strength_grouplasso);
};

#endif  // LIB_INCLUDE_TICK_HAWKES_INFERENCE_HAWKES_SUMGAUSSIANS_H_
