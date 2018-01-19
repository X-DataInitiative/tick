//
// Created by Martin Bompaire on 26/11/15.
//

#ifndef LIB_INCLUDE_TICK_HAWKES_SIMULATION_HAWKES_KERNELS_HAWKES_KERNEL_0_H_
#define LIB_INCLUDE_TICK_HAWKES_SIMULATION_HAWKES_KERNELS_HAWKES_KERNEL_0_H_

// License: BSD 3 clause

#include "hawkes_kernel.h"

/**
 * @class HawkesKernel0
 * @brief Hawkes zero kernel
 *
 * \f[
 *     \phi(t) = 0
 * \f]
 */
class HawkesKernel0 : public HawkesKernel {
 public :

  //! @brief Constructor
  HawkesKernel0() : HawkesKernel(0) {}

  /**
 * Computes L1 norm with an explicit formula
 * @param nsteps: number of steps for norm approximation (unused)
 * @return L1 norm of the kernel
 */
  double get_norm(int nsteps = 10000) override { return 0; }

  template<class Archive>
  void serialize(Archive &ar) {
    ar(cereal::make_nvp("HawkesKernel", cereal::base_class<HawkesKernel>(this)));
  }
};

typedef std::shared_ptr<HawkesKernel0> HawkesKernel0Ptr;

CEREAL_REGISTER_TYPE(HawkesKernel0);

#endif  // LIB_INCLUDE_TICK_HAWKES_SIMULATION_HAWKES_KERNELS_HAWKES_KERNEL_0_H_
