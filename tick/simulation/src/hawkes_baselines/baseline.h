
#ifndef TICK_SIMULATION_SRC_HAWKES_BASELINES_BASELINE_H_
#define TICK_SIMULATION_SRC_HAWKES_BASELINES_BASELINE_H_

#include <memory>
#include "base.h"

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

#endif  // TICK_SIMULATION_SRC_HAWKES_BASELINES_BASELINE_H_
