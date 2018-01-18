
//
// Created by Martin Bompaire on 26/11/15.
//

#ifndef LIB_INCLUDE_TICK_HAWKES_SIMULATION_HAWKES_KERNELS_HAWKES_KERNEL_POWER_LAW_H_
#define LIB_INCLUDE_TICK_HAWKES_SIMULATION_HAWKES_KERNELS_HAWKES_KERNEL_POWER_LAW_H_

// License: BSD 3 clause

#include "hawkes_kernel.h"

#include <cmath>

//
//
// The power law kernel
//
//      alpha*(x+delta)^{-beta} on [0,support]
//

/**
 * @class HawkesKernelPowerLaw
 * @brief Hawkes kernel for power law
 *
 * \f[
 *     \phi(t) = \alpha (\delta + t)^{- \beta} 1_{t > 0}
 * \f]
 *
 * Where \f$ \alpha \f$ is called the multiplier, \f$ \delta \f$ the cut-off and
 * \f$ \beta \f$ the exponent
 */
class HawkesKernelPowerLaw : public HawkesKernel {
 private:
  //! @brief multiplier of the kernel
  double multiplier;
  //! @brief exponent of the kernel
  double exponent;
  //! @brief cut-off of the kernel
  double cutoff;

  //! Getting the value of the kernel at the point x (where x is positive)
  double get_value_(double x) override;

 public :

  //! @brief simple getter
  double get_multiplier() { return multiplier; }
  //! @brief simple getter
  double get_exponent() { return exponent; }
  //! @brief simple getter
  double get_cutoff() { return cutoff; }

  /**
   * @brief Constructor
   * @param support: Support of the kernel (value after which the kernel equals zero)
   * @param error: If necessary, error will be used to compute the support
   * @note For specifying the support, you can either specify it directly (support > 0) or
   * specify an error such that support = f^{-1}(error)
   */
  HawkesKernelPowerLaw(double multiplier,
                       double cutoff,
                       double exponent,
                       double support = -1,
                       double error = 1e-5);

  //! @brief Copy constructor
  HawkesKernelPowerLaw(HawkesKernelPowerLaw &kernel) = default;

  /**
    * Empty constructor
    *
    * Used only for (de)serialization
  */
  HawkesKernelPowerLaw();

  /**
   * Computes L1 norm with an explicit formula
   * @param nsteps: number of steps for norm approximation (unused)
   * @return L1 norm of the kernel
   */
  double get_norm(int nsteps = 10000) override;

  template<class Archive>
  void serialize(Archive &ar) {
    ar(cereal::make_nvp("HawkesKernel", cereal::base_class<HawkesKernel>(this)));

    ar(CEREAL_NVP(multiplier));
    ar(CEREAL_NVP(exponent));
    ar(CEREAL_NVP(cutoff));
  }
};

CEREAL_REGISTER_TYPE(HawkesKernelPowerLaw);

#endif  // LIB_INCLUDE_TICK_HAWKES_SIMULATION_HAWKES_KERNELS_HAWKES_KERNEL_POWER_LAW_H_
