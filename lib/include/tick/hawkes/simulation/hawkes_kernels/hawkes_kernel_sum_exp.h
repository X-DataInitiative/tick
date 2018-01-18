
#ifndef LIB_INCLUDE_TICK_HAWKES_SIMULATION_HAWKES_KERNELS_HAWKES_KERNEL_SUM_EXP_H_
#define LIB_INCLUDE_TICK_HAWKES_SIMULATION_HAWKES_KERNELS_HAWKES_KERNEL_SUM_EXP_H_

// License: BSD 3 clause

#include <float.h>
#include "hawkes_kernel.h"

/**
 * @class HawkesKernelSumExp
 * @brief Hawkes kernel with sum of exponential decays
 *
 * \f[
 *     \phi(t) = \sum_{u=1}^{U} \alpha_u \beta_u \exp (- \beta_u t) 1_{t > 0}
 * \f]
 * where \f$ \alpha_u \f$ are the intensities of the kernel and \f$ \beta_u \f$ its decays.
 */
class HawkesKernelSumExp : public HawkesKernel {
  //! A static field to decide whether (approximated) fast formula for exponential
  //! should be used or not
  static bool use_fast_exp;

  //! Number of decays of the kernel, also noted \f$ U \f$
  ulong n_decays;

  //! Intensity of the kernel, also noted \f$ \alpha \f$
  ArrayDouble intensities;

  //! Decay of the kernel, also noted \f$ \alpha \f$
  ArrayDouble decays;

  // Used for efficiency for the computation of the convolution kernel*process(t)
  //! last time the convolution was computed
  double last_convolution_time;

  //! last value obtained for the convolution
  ArrayDouble last_convolution_values;

  //! last size of process is the last convolution computation
  ulong convolution_restart_index;

  //! Getting the value of the ith component of the kernel at the point x (where x is positive)
  inline double get_value_i(double x, ulong i);

  //! Getting the value of the kernel at the point x (where x is positive)
  double get_value_(double x) override;

  //! field telling if all intensities are positive. It is not a problem if some are negative
  //! except if we want to compute the future bound after a convolution.
  bool intensities_all_positive;

 public :

  /**
   * Constructor
   * @param intensities: Array of the intensities of the kernel
   * @param decay: Array of the decays of the kernel
   */
  explicit HawkesKernelSumExp(const ArrayDouble &intensities, const ArrayDouble &decays);

  /**
   * Copy constructor
   * @param kernel: kernel to be copied
   * @note this makes a copy of the given kernel's decays and intensitys arrays and
   * rewind the just created one
   */
  HawkesKernelSumExp(const HawkesKernelSumExp &kernel);

  HawkesKernelSumExp();

  /**
   * @brief Reset kernel for simulating a new realization
   * @note This is mandatory as soon as the process on which the convulution is done changes
   * or if the convolution is done up to a time which is older than the one the last
   * convolution
   */
  void rewind() override;

  /**
   * Computes L1 norm with an explicit formula
   * @param nsteps: number of steps for norm approximation (unused)
   * @return L1 norm of the kernel
   */
  double get_norm(int nsteps = 10000) override;

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
   */
  double get_convolution(const double time,
                         const ArrayDouble &timestamps,
                         double *const bound) override;

  //! simple setter
  static void set_fast_exp(bool flag) { use_fast_exp = flag; }
  //! simple getter
  static bool get_fast_exp() { return use_fast_exp; }

  /**
   * @brief Simple getter
   * @note Its makes a copy of the array to return it as a shared_ptr
   */
  SArrayDoublePtr get_intensities();

  /**
   * @brief Simple getter
   * @note Its makes a copy of the array to return it as a shared_ptr
   */
  SArrayDoublePtr get_decays();

  //! Simple getter
  ulong get_n_decays() { return n_decays; }

  /**
   * Get support for kernel plot
   * @return support end
   */
  double get_plot_support() override {
    return 3 / decays.min();
  }

  std::shared_ptr<HawkesKernel> duplicate_if_necessary(const std::shared_ptr<HawkesKernel> &kernel)
  override {
    return std::make_shared<HawkesKernelSumExp>(*this);
  }

  std::shared_ptr<HawkesKernel> duplicate_if_necessary() {
    return std::make_shared<HawkesKernelSumExp>(*this);
  }

  template<class Archive>
  void serialize(Archive &ar) {
    ar(cereal::make_nvp("HawkesKernel", cereal::base_class<HawkesKernel>(this)));

    ar(CEREAL_NVP(use_fast_exp));
    ar(CEREAL_NVP(n_decays));
    ar(CEREAL_NVP(intensities));
    ar(CEREAL_NVP(decays));
    ar(CEREAL_NVP(last_convolution_time));
    ar(CEREAL_NVP(last_convolution_values));
    ar(CEREAL_NVP(convolution_restart_index));
    ar(CEREAL_NVP(intensities_all_positive));
  }

 private:
  //! @brief Custom exponential function taking into account optimization level
  //! \param x : The value exponential is computed at
  inline double cexp(double x) {
    int optimization_level;
    if (use_fast_exp) {
      optimization_level = 1;
    } else {
      optimization_level = 0;
    }
    return optimized_exp(x, optimization_level);
  }
};

CEREAL_REGISTER_TYPE(HawkesKernelSumExp);

#endif  // LIB_INCLUDE_TICK_HAWKES_SIMULATION_HAWKES_KERNELS_HAWKES_KERNEL_SUM_EXP_H_
