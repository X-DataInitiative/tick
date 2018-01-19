
#ifndef LIB_INCLUDE_TICK_HAWKES_SIMULATION_HAWKES_KERNELS_HAWKES_KERNEL_TIME_FUNC_H_
#define LIB_INCLUDE_TICK_HAWKES_SIMULATION_HAWKES_KERNELS_HAWKES_KERNEL_TIME_FUNC_H_

// License: BSD 3 clause

#include "tick/base/time_func.h"
#include "hawkes_kernel.h"

/**
 * @class HawkesKernelTimeFunc
 * @brief Piecewise linear Hawkes kernels. This kernel is handled with a TimeFunction
 */
class HawkesKernelTimeFunc : public HawkesKernel {
 private:
  //! @brief The TimeFunction used by the kernel
  TimeFunction time_function;

  //! Getting the value of the kernel at the point x (where x is positive)
  double get_value_(double x) override;

 public :
  //! @brief Constructor
  explicit HawkesKernelTimeFunc(const TimeFunction &time_function);

  //! @brief Constructor
  HawkesKernelTimeFunc(const ArrayDouble &t_axis, const ArrayDouble &y_axis);

  //! @brief Copy constructor
  HawkesKernelTimeFunc(const HawkesKernelTimeFunc &kernel) = default;

  //! @brief Empty constructor
  HawkesKernelTimeFunc();

  /**
   * Returns the maximum of the kernel after time t
   * knowing that the value of the kernel at time t is value_at_t
  */
  double get_future_max(double t, double value_at_t) override;

  //! @brief simple getter
  const TimeFunction &get_time_function() const { return time_function; }

  template<class Archive>
  void serialize(Archive &ar) {
    ar(cereal::make_nvp("HawkesKernel", cereal::base_class<HawkesKernel>(this)));

    ar(CEREAL_NVP(time_function));
  }
};

CEREAL_REGISTER_TYPE(HawkesKernelTimeFunc);

#endif  // LIB_INCLUDE_TICK_HAWKES_SIMULATION_HAWKES_KERNELS_HAWKES_KERNEL_TIME_FUNC_H_
