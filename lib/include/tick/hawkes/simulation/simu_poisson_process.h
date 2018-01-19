
#ifndef LIB_INCLUDE_TICK_HAWKES_SIMULATION_SIMU_POISSON_PROCESS_H_
#define LIB_INCLUDE_TICK_HAWKES_SIMULATION_SIMU_POISSON_PROCESS_H_

// License: BSD 3 clause

#include "tick/base/time_func.h"
#include "simu_point_process.h"
#include <numeric>

/*! \class Poisson
 * \brief This is the class of constant Poisson processes
 */

class Poisson : public PP {
 public:
  /// @brief Process intensities
  SArrayDoublePtr intensities;

 public :
  /**
   * @brief A constructor for a 1 dimensional Poisson process
   * \param intensity : The intensity of the first and single dimension
   */
  explicit Poisson(double intensity, int seed = -1);

  /**
   * @brief Multi-dimensional constructor
   * \param intensities : Array of intensities
   */
  explicit Poisson(SArrayDoublePtr intensities, int seed = -1);

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

 public:
  /// @brief Returns the array of intensities
  SArrayDoublePtr get_intensities() { return intensities; }
};

#endif  // LIB_INCLUDE_TICK_HAWKES_SIMULATION_SIMU_POISSON_PROCESS_H_
