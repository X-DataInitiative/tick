
//
// Created by Martin Bompaire on 26/11/15.
//

#ifndef LIB_INCLUDE_TICK_HAWKES_SIMULATION_HAWKES_KERNELS_HAWKES_KERNEL_EXP_H_
#define LIB_INCLUDE_TICK_HAWKES_SIMULATION_HAWKES_KERNELS_HAWKES_KERNEL_EXP_H_

// License: BSD 3 clause

#include "tick/base/base.h"
#include "hawkes_kernel.h"

/**
 * @class HawkesKernelExp
 * @brief Hawkes kernel with exponential decay
 *
 * \f[
 *     \phi(t) = \alpha \beta \exp (- \beta t) 1_{t > 0}
 * \f]
 * where \f$ \alpha \f$ is the intensity of the kernel and \f$ \beta \f$ its decay.
 */
class HawkesKernelExp : public HawkesKernel {
  //! A static field to decide whether (approximated) fast formula for exponential
  //! should be used or not
  static bool use_fast_exp;

  //! Intensity of the kernel, also noted \f$ \alpha \f$
  double intensity;

  //! Decay of the kernel, also noted \f$ \alpha \f$
  double decay;

  // Used for efficiency for the computation of the convolution kernel*process(t)
  //! last time the convolution was computed
  double last_convolution_time;

  //! last value obtained for the convolution
  double last_convolution_value;

  //! last size of process is the last convolution computation
  ulong convolution_restart_index;

  //! Getting the value of the kernel at the point x (where x is positive)
  double get_value_(double x) override;

 public :

  /**
   * Constructor
   * @param intensity: intensity of the kernel
   * @param decay: decay of the kernel
   */
  HawkesKernelExp(double intensity, double decay);

  /**
   * Copy constructor
   * @param kernel: kernel to be copied
   * @note this copies the given kernel's decay and intensity and rewind the just created one
   */
  HawkesKernelExp(const HawkesKernelExp &kernel);

  /**
   * Empty constructor
   *
   * Used only for (de)serialization
   */
  HawkesKernelExp();

  /**
   * @brief Reset kernel for simulating a new realization
   * @note This is mandatory as soon as the process on which the convolution is done changes
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

  //! simple getter
  double get_intensity() { return intensity; }
  //! simple getter
  double get_decay() { return decay; }

  /**
   * Get support for kernel plot
   * @return support end
   */
  double get_plot_support() override {
    return 3 / decay;
  }

  std::shared_ptr<HawkesKernel> duplicate_if_necessary(
    const std::shared_ptr<HawkesKernel> &kernel) override {
    return std::make_shared<HawkesKernelExp>(*this);
  }

  std::shared_ptr<HawkesKernel> duplicate_if_necessary() {
    return std::make_shared<HawkesKernelExp>(*this);
  }

  template<class Archive>
  void serialize(Archive &ar) {
    ar(cereal::make_nvp("HawkesKernel", cereal::base_class<HawkesKernel>(this)));

    ar(CEREAL_NVP(use_fast_exp));
    ar(CEREAL_NVP(intensity));
    ar(CEREAL_NVP(decay));
    ar(CEREAL_NVP(last_convolution_time));
    ar(CEREAL_NVP(last_convolution_value));
    ar(CEREAL_NVP(convolution_restart_index));
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

CEREAL_REGISTER_TYPE(HawkesKernelExp);

#endif  // LIB_INCLUDE_TICK_HAWKES_SIMULATION_HAWKES_KERNELS_HAWKES_KERNEL_EXP_H_
