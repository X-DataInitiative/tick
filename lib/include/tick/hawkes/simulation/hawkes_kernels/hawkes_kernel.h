
#ifndef LIB_INCLUDE_TICK_HAWKES_SIMULATION_HAWKES_KERNELS_HAWKES_KERNEL_H_
#define LIB_INCLUDE_TICK_HAWKES_SIMULATION_HAWKES_KERNELS_HAWKES_KERNEL_H_

// License: BSD 3 clause

#include "tick/base/defs.h"
#include "tick/array/sarray.h"

#include <memory>

#include <cereal/types/polymorphic.hpp>
#include <cereal/types/base_class.hpp>

/**
 * @class HawkesKernel
 *  The kernel class allows to define 1 element of the kernel matrix of a Hawkes process
 */
class HawkesKernel {
 protected:
  /**
   * Support is used for dealing with support of the kernel
   * (the kernel is 0 outside of [0, support])
   *
   * The constructor of the Kernel class should ALWAYS set the field support.
   * There are two cases
   *   - support == 0 : that means that the kernel is a zero-kernel
   *   - support > 0  : the kernel is 0 outside of [0,support]
   */
  double support;

  /**
   * Getting the value of the kernel at the point x
   * This is the function to be overloaded. It is called by the method get_value when
   * \f$ x \in [0, support] \f$
   */
  virtual double get_value_(double x) { return 0; }

 public:
  //! @brief Reset kernel for simulating a new realization
  virtual void rewind() {}

  //! @brief Reset kernel for simulating a new realization
  explicit HawkesKernel(double support = 0);

  //! @brief Copy constructor
  HawkesKernel(const HawkesKernel &kernel);

  // Some kernels cannot be shared, so this function is called before a kernel is used
  // TODO(martin) change function
  virtual std::shared_ptr<HawkesKernel> duplicate_if_necessary(
    const std::shared_ptr<HawkesKernel> &kernel) {
    return kernel;
  }

  //! @brief Returns if this kernel is equal to 0
  bool is_zero() const { return support <= 0; }

  //! @brief Returns the upperbound of the support
  double get_support() const { return support; }

  //! @brief Returns the value of the kernel at x
  double get_value(double x);

  //! @brief Returns the value of the kernel for each t in t_values
  SArrayDoublePtr get_values(const ArrayDouble &t_values);

  /**
   * Computes L1 norm
   * @param nsteps: number of steps used for integral discretization
   * @note By default it approximates Riemann sum with step-wise function. It should be
   * overloaded if L1 norm closed formula exists
   */
  virtual double get_norm(int nsteps = 10000);

  /**
   * Computes the convolution of the process with the kernel
   * \f[
   *     \int_0^t \phi(t - s) dN(s) = \sum_{t_k} \phi(t - t_k)
   * \f]
   * @param time: The time \f$ t \f$ up to the convolution is computed
   * @param timestamps: The process \f$ N \f$ with which the convolution is computed
   * @param bound: if `bound != nullptr` we store in this variable we store the maximum value that
   * the convolution can reach until next jump. This is useful for Ogata's thinning algorithm.
   * @return the value of the convolution
   * @note Should be overloaded for efficiency if there is a faster way to compute
   * this convolution than just regular algorithm
   */
  virtual double get_convolution(const double time,
                                 const ArrayDouble &timestamps,
                                 double *const bound);

  /**
   * Returns the maximum of the kernel after time t
   * knowing that the value of the kernel at time t is value_at_t
   * @note default is value_at_t (decreasing kernel)
   */
  virtual double get_future_max(double t, double value_at_t) {
    return value_at_t;
  }

  //! Returns support used to plot the kernel
  virtual double get_plot_support() { return get_support(); }

  template<class Archive>
  void serialize(Archive &ar) {
    ar(CEREAL_NVP(support));
  }
};

// A shared pointer to the HawkesKernel class
typedef std::shared_ptr<HawkesKernel> HawkesKernelPtr;

#endif  // LIB_INCLUDE_TICK_HAWKES_SIMULATION_HAWKES_KERNELS_HAWKES_KERNEL_H_
