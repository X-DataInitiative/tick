
#ifndef LIB_INCLUDE_TICK_HAWKES_SIMULATION_HAWKES_BASELINES_BASELINE_H_
#define LIB_INCLUDE_TICK_HAWKES_SIMULATION_HAWKES_BASELINES_BASELINE_H_

// License: BSD 3 clause

#include "tick/base/base.h"

#include <memory>

/*! \class HawkesBaseline
 * \brief A abstract class for Hawkes baselines
 */
class HawkesBaseline {
 public:
  //! @brief Empty constuctor
  HawkesBaseline() {}

  //! @brief get value of the baseline at time t
  virtual double get_value(double t) = 0;

  //! @brief get value of the baseline at times t
  virtual SArrayDoublePtr get_value(ArrayDouble &t) = 0;

  //! @brief get the future maximum reachable value of the baseline after time t
  virtual double get_future_bound(double t) = 0;

  template<class Archive>
  void serialize(Archive &ar) {}
};

typedef std::shared_ptr<HawkesBaseline> HawkesBaselinePtr;

#endif  // LIB_INCLUDE_TICK_HAWKES_SIMULATION_HAWKES_BASELINES_BASELINE_H_
