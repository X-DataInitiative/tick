
#ifndef LIB_INCLUDE_TICK_HAWKES_INFERENCE_HAWKES_EM_H_
#define LIB_INCLUDE_TICK_HAWKES_INFERENCE_HAWKES_EM_H_

// License: BSD 3 clause

#include "tick/base/base.h"
#include "tick/hawkes/model/base/model_hawkes_list.h"

////////////////////////////////////////////////////////////////////////////////////////////
//
//
// The following class implements the n-dimensional version of the 1-dimensional algorithm
// presented in the paper `A Nonparametric EM algorithm for Multiscale Hawkes Processes`
// by Lewis and Molher, 2011
//
// The implementation is detailed in tex/HawkesEM.tex
//
////////////////////////////////////////////////////////////////////////////////////////////

class HawkesEM : public ModelHawkesList {
  //! @brief Maximum support of the kernel
  double kernel_support;

  //! @brief Number of discretizations of the kernel
  ulong kernel_size;

  //! @brief explicit discretization of the kernel
  SArrayDoublePtr kernel_discretization;

  //! @brief buffer variables
  ArrayDouble2d next_mu;
  ArrayDouble2d next_kernels;
  ArrayDouble2d unnormalized_kernels;

 public :
  HawkesEM(const double kernel_support, const ulong kernel_size,
           const int max_n_threads = 1);

  explicit HawkesEM(const SArrayDoublePtr kernel_discretization, const int max_n_threads = 1);

  //! @brief allocate buffer arrays once data has been given
  void allocate_weights();

  //! @brief The main method to perform one iteration
  void solve(ArrayDouble &mu, ArrayDouble2d &kernels);

  //! @brief Compute loglikelihood of a given kernel and baseline
  double loglikelihood(const ArrayDouble &mu, ArrayDouble2d &kernels);

  SArrayDouble2dPtr get_kernel_norms(ArrayDouble2d &kernels) const;

  double get_kernel_support() const { return kernel_support; }

  ulong get_kernel_size() const { return kernel_size; }

  double get_kernel_fixed_dt() const;

  SArrayDoublePtr get_kernel_discretization() const;

  //! @brief set kernel support
  void set_kernel_support(const double kernel_support);

  //! @brief set kernel size
  void set_kernel_size(const ulong kernel_size);

  //! @brief set kernel_dt by adjust `kernel_size` accordingly
  //! such that kernel_support / kernel_size ~= kernel_dt
  void set_kernel_dt(const double kernel_dt);

  void set_kernel_discretization(const SArrayDoublePtr kernel_discretization);

 private:
  //! @brief A method called in parallel by the method 'solve'
  //! @param r_u : r * n_realizations + u, tells which realization and which node
  void solve_ur(const ulong r_u, const ArrayDouble &mu, ArrayDouble2d &kernel);

  //! @brief A method called in parallel by the method 'loglikelihood'
  //! @param r_u : r * n_realizations + u, tells which realization and which node
  double loglikelihood_ur(const ulong r_u, const ArrayDouble &mu, ArrayDouble2d &kernels);

  //! @brief A method called by solve_ur and logliklihood_ur to compute all intensities at
  //! all timestamps occuring in node u of realization r
  //! @param r_u : r * n_realizations + u, tells which realization and which node
  //! @param intensity_func : function that will be called for all timestamps with the intensity at
  //! this timestamp as argument
  //! @param store_unnormalized_kernel : solve_ur method needs to store an unnormalized version
  //! of the kernels in the class variable unnormalized_kernels
  void compute_intensities_ur(const ulong r_u, const ArrayDouble &mu, ArrayDouble2d &kernels,
                              std::function<void(double)> intensity_func,
                              bool store_unnormalized_kernel);

  double compute_compensator_ur(const ulong r_u, const ArrayDouble &mu, ArrayDouble2d &kernels);

  void check_baseline_and_kernels(const ArrayDouble &mu, ArrayDouble2d &kernels) const;

  //! @brief Discretization parameter of the kernel
  //! If kernel_discretization is a nullptr then it is equal to kernel_support / kernel_size
  //! otherwise it is equal to the difference of
  //! kernel_discretization[m+1] - kernel_discretization[m]
  double get_kernel_dt(const ulong m = 0) const;
};

#endif  // LIB_INCLUDE_TICK_HAWKES_INFERENCE_HAWKES_EM_H_
