
#ifndef LIB_INCLUDE_TICK_HAWKES_MODEL_BASE_MODEL_HAWKES_LEASTSQ_H_
#define LIB_INCLUDE_TICK_HAWKES_MODEL_BASE_MODEL_HAWKES_LEASTSQ_H_

// License: BSD 3 clause

#include "tick/base/base.h"
#include "tick/hawkes/model/base/model_hawkes_list.h"
#include "tick/hawkes/model/base/model_hawkes_single.h"

/** \class ModelHawkesLeastSq
 * \brief Base class of Hawkes models handling several realizations
 */
class DLL_PUBLIC ModelHawkesLeastSq : public ModelHawkesList {
 protected:
  //! @brief Flag telling if precompations arrays have been allocated or not
  bool weights_allocated;

  //! @bbrief aggregated model used to compute loss, gradient and hessian
  std::unique_ptr<ModelHawkesSingle> aggregated_model;

 public:
  //! @brief Constructor
  //! \param max_n_threads : number of cores to be used for multithreading. If negative,
  //! the number of physical cores will be used
  //! \param optimization_level : 0 corresponds to no optimization and 1 to use of faster (approximated) exponential function
  ModelHawkesLeastSq(const int max_n_threads = 1,
                     const unsigned int optimization_level = 0);

  void incremental_set_data(const SArrayDoublePtrList1D &timestamps, double end_time);

  /**
   * @brief Precomputations of intermediate values
   * They will be used to compute faster loss and gradient
   */
  void compute_weights();

  /**
   * @brief Compute loss
   * \param coeffs : Point in which loss is computed
   * \return Loss' value
   */
  double loss(const ArrayDouble &coeffs) override;

  /**
   * @brief Compute loss corresponding to sample i (between 0 and rand_max = dim)
   * \param i : selected dimension
   * \param coeffs : Point in which loss is computed
   * \return Loss' value
   */
  double loss_i(const ulong i, const ArrayDouble &coeffs) override;

  /**
   * @brief Compute gradient
   * \param coeffs : Point in which gradient is computed
   * \param out : Array in which the value of the gradient is stored
   */
  void grad(const ArrayDouble &coeffs, ArrayDouble &out) override;

  /**
   * @brief Compute gradient corresponding to sample i (between 0 and rand_max = dim)
   * \param i : selected dimension
   * \param coeffs : Point in which gradient is computed
   * \param out : Array in which the value of the gradient is stored
   */
  void grad_i(const ulong i, const ArrayDouble &coeffs, ArrayDouble &out) override;

 protected:
  //! @brief allocate arrays to store precomputations
  virtual void allocate_weights() {}

  //! @brief synchronize aggregate_model with this instance
  virtual void synchronize_aggregated_model() {}

  virtual void compute_weights_timestamps_list() {}
  virtual void compute_weights_timestamps(const SArrayDoublePtrList1D &timestamps,
                                          double end_time) {}

 public:
  template<class Archive>
  void serialize(Archive &ar) {
    ar(cereal::make_nvp("ModelHawkesList", cereal::base_class<ModelHawkesList>(this)));

    ar(CEREAL_NVP(weights_allocated));
    ar(CEREAL_NVP(aggregated_model));
  }
};

#endif  // LIB_INCLUDE_TICK_HAWKES_MODEL_BASE_MODEL_HAWKES_LEASTSQ_H_
