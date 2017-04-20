//
// Created by Martin Bompaire on 02/06/15.
//

#ifndef TICK_SIMULATION_SRC_HAWKES_H_
#define TICK_SIMULATION_SRC_HAWKES_H_

#include <memory>
#include "time_func.h"
#include <float.h>

#include "hawkes_kernels/hawkes_kernel.h"
#include "hawkes_kernels/hawkes_kernel_0.h"
#include "hawkes_kernels/hawkes_kernel_exp.h"
#include "hawkes_kernels/hawkes_kernel_power_law.h"
#include "hawkes_kernels/hawkes_kernel_sum_exp.h"
#include "hawkes_kernels/hawkes_kernel_time_func.h"

#include "varray.h"
#include "pp.h"

//*********************************************************************************
//
// The mu class allows to define 1 element of the mu vector
//
//*********************************************************************************


// This is a constant mu
/*! \class HawkesMu
 * \brief A basic wrapper of double to represent \f$ \mu \f$ of Hawkes processes
 */
class HawkesMu {
  // The value
  double mu;

 public:
  /**
   * @brief constructor that takes a double and wrap it in a HawkesMu
   * \param mu : The value of \f$ \mu \f$
   */
  explicit HawkesMu(double mu = 0) { this->mu = mu; }

  /// @brief getter for \f$ \mu \f$
  virtual double get_value() { return mu; }

  template<class Archive>
  void serialize(Archive &ar) {
    ar(CEREAL_NVP(mu));
  }
};

typedef std::shared_ptr<HawkesMu> HawkesMuPtr;


//*********************************************************************************
//
// The Hawkes class
//
//*********************************************************************************


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
  std::vector<HawkesMuPtr> mus;

 public :
  /**
   * @brief A constructor for an empty multidimensional Hawkes process
   * \param dimension : The dimension of the Hawkes process
   */
  explicit Hawkes(unsigned int dimension, int seed = -1);

  // This forbids the unwanted copy of an Hawkes process
  Hawkes(Hawkes &hawkes) = delete;

  ~Hawkes();

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
   * @brief Set mu for a specific dimension
   * \param i : the dimension
   * \param mu : the HawkesMu to be set
   */
  void set_mu(unsigned int i, const HawkesMuPtr &mu);

  /**
   * @brief Set mu for a specific dimension
   * \param i : the dimension
   * \param mu : a double that will be used to construct a HawkesMu
   */
  void set_mu(unsigned int i, double mu);

  /**
   * @brief Get mu for a specific dimension
   * \param i : the dimension
   */
  double get_mu(unsigned int i);

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

 public:
  template<class Archive>
  void serialize(Archive &ar) {
    ar(cereal::make_nvp("PP", cereal::base_class<PP>(this)));

    ar(CEREAL_NVP(mus));
    ar(CEREAL_NVP(kernels));
  }
};

CEREAL_SPECIALIZE_FOR_ALL_ARCHIVES(Hawkes, cereal::specialization::member_serialize)

#endif  // TICK_SIMULATION_SRC_HAWKES_H_
