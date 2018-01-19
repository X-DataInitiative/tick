
#ifndef LIB_INCLUDE_TICK_HAWKES_INFERENCE_HAWKES_BASIS_KERNELS_H_
#define LIB_INCLUDE_TICK_HAWKES_INFERENCE_HAWKES_BASIS_KERNELS_H_

// License: BSD 3 clause

#include "tick/base/base.h"
#include "tick/hawkes/model/base/model_hawkes_list.h"


////////////////////////////////////////////////////////////////////////////////////////////
//
//
// The following class implements the algorithm described in the paper
// `Learning Triggering Kernels for Multi-dimensional Hawkes Processes` by
// Zhou, Zha, and Song (2013) in Proc of the International Conf. on Machine Learning.
// Some rewriting notes for implementing the algorithm can be found in the file
// hawkes_basis_kernels.pdf in the directory tex/hawkes_basis_kernels (LaTeX)
//
//
////////////////////////////////////////////////////////////////////////////////////////////

class HawkesBasisKernels : public ModelHawkesList {
  //! @brief Maximum support of the kernel
  double kernel_support;

  //! @brief Number of discretizations of the kernel
  ulong kernel_size;

  //! @brief Number of basis functions used to generate kernels
  ulong n_basis;

  //! @brief penalty parameter
  double alpha;

  //! @brief Buffer variables
  ArrayDouble2d rud, Dudm, Dudm_temp, Cudm, Gdm, a_sum_vd;
  ArrayDouble2d quvd, quvd_temp;

 public :
  HawkesBasisKernels(const double kernel_support,
                     const ulong kernel_size,
                     const ulong n_basis,
                     const double alpha,
                     const int max_n_threads = 1);

  double solve(ArrayDouble &mu,
               ArrayDouble2d &gdm,
               ArrayDouble2d &auvd,
               ulong max_iter_gdm,
               double max_tol_gdm);

 private:
  void solve_u(ulong u,
               ArrayDouble &mu,
               ArrayDouble2d &gdm,
               ArrayDouble2d &auvd);

  void allocate_weights();

 public:
  //! @brief we need to override this function as we do not parallelize over realizations
  unsigned int get_n_threads() const override;

  double get_kernel_support() const { return kernel_support; }

  ulong get_kernel_size() const { return kernel_size; }

  inline double get_kernel_dt() const { return kernel_support / kernel_size; }

  //! @brief if n_basis was not set, we use n_nodes
  inline ulong get_n_basis() const {
    return n_basis == 0 ? n_nodes : n_basis;
  }

  double get_alpha() const { return alpha; }

  SArrayDoublePtr get_kernel_discretization() const {
    ArrayDouble kernel_discretization_tmp = arange<double>(0, kernel_size + 1);
    kernel_discretization_tmp.mult_fill(kernel_discretization_tmp, get_kernel_dt());
    return kernel_discretization_tmp.as_sarray_ptr();
  }

  //! @brief set kernel support
  void set_kernel_support(const double kernel_support);

  //! @brief set kernel size
  void set_kernel_size(const ulong kernel_size);

  //! @brief set kernel_dt by adjust `kernel_size` accordingly
  //! such that kernel_support / kernel_size ~= kernel_dt
  void set_kernel_dt(const double kernel_dt);

  //! @brief set number of kernel basis
  void set_n_basis(const ulong n_basis);

  //! @brief set kernel alpha (penalization parameter)
  void set_alpha(const double alpha);
};

#endif  // LIB_INCLUDE_TICK_HAWKES_INFERENCE_HAWKES_BASIS_KERNELS_H_
