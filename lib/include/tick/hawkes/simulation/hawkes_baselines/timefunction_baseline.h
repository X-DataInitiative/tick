
#ifndef LIB_INCLUDE_TICK_HAWKES_SIMULATION_HAWKES_BASELINES_TIMEFUNCTION_BASELINE_H_
#define LIB_INCLUDE_TICK_HAWKES_SIMULATION_HAWKES_BASELINES_TIMEFUNCTION_BASELINE_H_

// License: BSD 3 clause

#include "tick/base/base.h"
#include "baseline.h"

#include <cereal/types/polymorphic.hpp>

/*! \class HawkesBaseline
 * \brief Class of time varying baselines modeled by TimeFunction
 */
class HawkesTimeFunctionBaseline : public HawkesBaseline {
  //! @brief The timefunction that will compute time varying baseline
  TimeFunction time_function;

 public:
  //! @brief default constructor (0 baseline)
  HawkesTimeFunctionBaseline();

  //! @brief TimeFunction constructor
  explicit HawkesTimeFunctionBaseline(TimeFunction time_function);

  /**
   * @brief constructor that takes a double and wrap it in a HawkesBaseline
   * \param times : The changing times of the baseline
   * \param values : The values of \f$ \mu \f$
   */
  HawkesTimeFunctionBaseline(ArrayDouble &times, ArrayDouble &values);

  //! @brief get value of the baseline at time t
  double get_value(double t) override;

  //! @brief get value of the baseline at time t
  SArrayDoublePtr get_value(ArrayDouble &t) override;

  //! @brief get the future maximum reachable value of the baseline after time t
  double get_future_bound(double t) override;

  template<class Archive>
  void serialize(Archive &ar) {
    ar(cereal::make_nvp("HawkesBaseline", cereal::base_class<HawkesBaseline>(this)));
    ar(CEREAL_NVP(time_function));
  }
};

CEREAL_REGISTER_TYPE(HawkesTimeFunctionBaseline);

#endif  // LIB_INCLUDE_TICK_HAWKES_SIMULATION_HAWKES_BASELINES_TIMEFUNCTION_BASELINE_H_
