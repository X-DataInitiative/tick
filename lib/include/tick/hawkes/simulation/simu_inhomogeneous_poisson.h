//
// Created by Martin Bompaire on 24/11/15.
//

#ifndef LIB_INCLUDE_TICK_HAWKES_SIMULATION_SIMU_INHOMOGENEOUS_POISSON_H_
#define LIB_INCLUDE_TICK_HAWKES_SIMULATION_SIMU_INHOMOGENEOUS_POISSON_H_

// License: BSD 3 clause

#include "tick/base/time_func.h"
#include "simu_point_process.h"

/*! \class InhomogeneousPoisson
 * \brief This is the class for Poisson processes with variable intensities
 *
 * Their intensities are modeled by TimeFunction
 */

class InhomogeneousPoisson : public PP {
  std::vector<TimeFunction> intensities_functions;
 public :
  /**
  * @brief A constructor for a 1 dimensional inhomogeneous Poisson process
  * \param intensities_function : The intensity function of the first and
   * single dimension
  */
  explicit InhomogeneousPoisson(const TimeFunction &intensities_function, int seed = -1);

  /**
   * @brief Multi-dimensional constructor
   * \param intensities_functions : Array of intensity functions
   */
  explicit InhomogeneousPoisson(const std::vector<TimeFunction> &intensities_functions,
                                int seed = -1);

  /**
   * @brief Returns for a given dimension, the value of the intensity
   * function at different times
   * \param dimension : The selected dimension
   * \param times_values : the times at which intensity will be evaluated
   */
  // TODO: returns intensities as TimeFunction
  SArrayDoublePtr intensity_value(int dimension, ArrayDouble &times_values) {
    return intensities_functions[dimension].value(times_values);
  }

 private:
  /**
   * @brief Virtual method called once (at startup) to set the initial
   * intensity
   * \param intensity : The intensity vector (of size #dimension) to initialize
   * \param total_intensity_bound : A pointer to the variable that will hold a
   * bound of future total intensity
   */
  virtual void init_intensity_(ArrayDouble &intensity,
                               double *total_intensity_bound);

  /**
   * @brief Updates the current time so that it goes forward of delay seconds
   * The intensities must be updated and track recorded if needed
   * Returns false if negative intensities were encountered
   * \param delay : Time to update
   * \param intensity : The intensity vector to update
   * \param total_intensity_bound : If not NULL then used to set a bound of
   * total future intensity
   */
  virtual bool update_time_shift_(double delay,
                                  ArrayDouble &intensity,
                                  double *total_intensity_bound);
};

#endif  // LIB_INCLUDE_TICK_HAWKES_SIMULATION_SIMU_INHOMOGENEOUS_POISSON_H_
