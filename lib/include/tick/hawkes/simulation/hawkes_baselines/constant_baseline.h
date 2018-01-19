
#ifndef LIB_INCLUDE_TICK_HAWKES_SIMULATION_HAWKES_BASELINES_CONSTANT_BASELINE_H_
#define LIB_INCLUDE_TICK_HAWKES_SIMULATION_HAWKES_BASELINES_CONSTANT_BASELINE_H_

// License: BSD 3 clause

#include "baseline.h"

#include <cereal/types/polymorphic.hpp>

/*! \class HawkesConstantBaseline
 * \brief A basic wrapper of double to represent \f$ \mu \f$ of Hawkes processes
 */
class HawkesConstantBaseline : public HawkesBaseline {
  //! @brief The value
  double value;

 public:
  /**
   * @brief constructor that takes a double and wrap it in a HawkesBaseline
   * \param value : The value of \f$ \mu \f$
   */
  explicit HawkesConstantBaseline(double value = 0);

  //! @brief get value of the baseline at time t
  double get_value(double t) override;

  //! @brief get value of the baseline at times t
  SArrayDoublePtr get_value(ArrayDouble &t) override;

  //! @brief get the future maximum reachable value of the baseline after time t
  double get_future_bound(double t) override;

  template<class Archive>
  void serialize(Archive &ar) {
    ar(cereal::make_nvp("HawkesBaseline", cereal::base_class<HawkesBaseline>(this)));
    ar(CEREAL_NVP(value));
  }
};

CEREAL_REGISTER_TYPE(HawkesConstantBaseline);

#endif  // LIB_INCLUDE_TICK_HAWKES_SIMULATION_HAWKES_BASELINES_CONSTANT_BASELINE_H_
