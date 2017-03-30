
#ifndef TICK_INFERENCE_SRC_HAWKES_EM_H_
#define TICK_INFERENCE_SRC_HAWKES_EM_H_

#include "base.h"
#include "base/hawkes_list.h"

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

 private:
  //! @brief A method called in parallel by the method 'solve'
  //! @param r_u : r * n_realizations + u, tells which realization and which node
  void solve_u_r(const ulong r_u, const ArrayDouble &mu, ArrayDouble2d &kernel);

  //! @brief Discretization parameter of the kernel
  //! If kernel_discretization is a nullptr then it is equal to kernel_support / kernel_size
  //! otherwise it is equal to the difference of
  //! kernel_discretization[m+1] - kernel_discretization[m]
  inline double get_kernel_dt(const ulong m = 0) const {
    if (kernel_discretization == nullptr) {
      return kernel_support / kernel_size;
    } else {
      return (*kernel_discretization)[m + 1] - (*kernel_discretization)[m];
    }
  }

 public:
  double get_kernel_support() const { return kernel_support; }

  ulong get_kernel_size() const { return kernel_size; }

  double get_kernel_fixed_dt() const;

  SArrayDoublePtr get_kernel_discretization() const {
    if (kernel_discretization == nullptr) {
      ArrayDouble kernel_discretization_tmp = arange<double>(0, kernel_size + 1);
      kernel_discretization_tmp.mult_fill(kernel_discretization_tmp, get_kernel_fixed_dt());
      return kernel_discretization_tmp.as_sarray_ptr();
    } else {
      return kernel_discretization;
    }
  }

  //! @brief set kernel support
  void set_kernel_support(const double kernel_support);

  //! @brief set kernel size
  void set_kernel_size(const ulong kernel_size);

  //! @brief set kernel_dt by adjust `kernel_size` accordingly
  //! such that kernel_support / kernel_size ~= kernel_dt
  void set_kernel_dt(const double kernel_dt);

  void set_kernel_discretization(const SArrayDoublePtr kernel_discretization);
};

#endif  // TICK_INFERENCE_SRC_HAWKES_EM_H_
