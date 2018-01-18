//
// Created by Martin Bompaire on 02/06/15.
//

#ifndef LIB_INCLUDE_TICK_HAWKES_SIMULATION_SIMU_HAWKES_H_
#define LIB_INCLUDE_TICK_HAWKES_SIMULATION_SIMU_HAWKES_H_

// License: BSD 3 clause

#include "tick/base/defs.h"

#include <memory>

#include <cfloat>
#include <memory>

#include "tick/base/time_func.h"
#include "tick/array/varray.h"
#include "simu_point_process.h"

#include "hawkes_baselines/baseline.h"
#include "hawkes_baselines/constant_baseline.h"
#include "hawkes_baselines/timefunction_baseline.h"

#include "hawkes_kernels/hawkes_kernel.h"
#include "hawkes_kernels/hawkes_kernel_0.h"
#include "hawkes_kernels/hawkes_kernel_exp.h"
#include "hawkes_kernels/hawkes_kernel_power_law.h"
#include "hawkes_kernels/hawkes_kernel_sum_exp.h"
#include "hawkes_kernels/hawkes_kernel_time_func.h"

/*! \class Hawkes
 * \brief This class stands for all types of Hawkes
 * processes
 *
 * They are defined by the intensity:
 * \f[
 *     \lambda = \mu + \phi * dN
 * \f]
 * where
 *   - \f$ \phi \f$ are the kernels
 *   - \f$ dN \f$ are the processes differentiates
 *   - \f$ * \f$ is a convolution product
 */
class Hawkes : public PP {
 public:
  /// @brief The kernel matrix
  std::vector<HawkesKernelPtr> kernels;

  /// @brief The mus
  std::vector<HawkesBaselinePtr> baselines;

 public :
  /**
   * @brief A constructor for an empty multidimensional Hawkes process
   * \param n_nodes : The dimension of the Hawkes process
   */
  explicit Hawkes(unsigned int n_nodes, int seed = -1);

  // This forbids the unwanted copy of an Hawkes process
  Hawkes(Hawkes &hawkes) = delete;

 public:
  virtual void reset();

  /**
   * @brief Set kernel for a specific row and column
   * \param i : the row
   * \param j : the column
   * \param kernel : the kernel to be stored
   * \note This will do a hard copy of the kernel if and only if this
   * kernel has its own memory (e.g HawkesKernelExp), otherwise we will only
   * share a pointer to this kernel.
   */
  void set_kernel(unsigned int i, unsigned int j, HawkesKernelPtr &kernel);

  /**
   * @brief Get kernel for a specific row and column
   * \param i : the row
   * \param j : the column
   */
  HawkesKernelPtr get_kernel(unsigned int i, unsigned int j);

  /**
   * @brief Set baseline for a specific dimension
   * \param i : the dimension
   * \param baseline : a double that will be used to construct a HawkesConstantBaseline
   */
  void set_baseline(unsigned int i, double baseline);

  /**
   * @brief Set baseline for a specific dimension
   * \param i : the dimension
   * \param time_function : a TimeFunction that will be used to construct a
   * HawkesTimeFunctionBaseline
   */
  void set_baseline(unsigned int i, TimeFunction time_function);

  /**
  * @brief Set baseline for a specific dimension
  * \param i : the dimension
  * \param times : times that will be used to construct a HawkesTimeFunctionBaseline
  * \param values : values that will be used to construct a HawkesTimeFunctionBaseline
  */
  void set_baseline(unsigned int i, ArrayDouble &times, ArrayDouble &values);

  /**
   * @brief Get baseline for a specific dimension at a given time
   * \param i : the dimension
   * \param t : considered time
   */
  double get_baseline(unsigned int i, double t);

  /**
   * @brief Get baseline for a specific dimension at a given time
   * \param i : the dimension
   * \param t : considered times
   */
  SArrayDoublePtr get_baseline(unsigned int i, ArrayDouble &t);

 private :
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

  /**
   * @brief Get future baseline maximum reachable value for a specific dimension at a given time
   * \param i : the dimension
   * \param t : considered time
   */
  double get_baseline_bound(unsigned int i, double t);

  /**
   * @brief Set baseline for a specific dimension
   * \param i : the dimension
   * \param baseline : the HawkesBaseline to be set
   */
  void set_baseline(unsigned int i, const HawkesBaselinePtr &baseline);

 public:
  template<class Archive>
  void serialize(Archive &ar) {
    ar(cereal::make_nvp("PP", cereal::base_class<PP>(this)));

    ar(CEREAL_NVP(baselines));
    ar(CEREAL_NVP(kernels));
  }
};

CEREAL_SPECIALIZE_FOR_ALL_ARCHIVES(Hawkes, cereal::specialization::member_serialize)

#endif  // LIB_INCLUDE_TICK_HAWKES_SIMULATION_SIMU_HAWKES_H_
