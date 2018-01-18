
#ifndef LIB_INCLUDE_TICK_HAWKES_MODEL_BASE_MODEL_HAWKES_LOGLIK_SINGLE_H_
#define LIB_INCLUDE_TICK_HAWKES_MODEL_BASE_MODEL_HAWKES_LOGLIK_SINGLE_H_

// License: BSD 3 clause

#include "tick/base/base.h"

#include "tick/hawkes/model/base/model_hawkes_single.h"

class ModelHawkesLogLik;

/**
 * \class ModelHawkesLogLikSingle
 * \brief Class for computing loglikelihood function and gradient for Hawkes processes with
 * exponential kernels with fixed exponent (i.e., \f$ \alpha \beta e^{-\beta t} \f$, with fixed
 * decay)
 */
class DLL_PUBLIC ModelHawkesLogLikSingle : public ModelHawkesSingle {
 protected:
  // Some arrays used for intermediate computings. They are initialized in init()
  //! @brief kernel intensity of node j on node i at time t_i_k
  ArrayDouble2dList1D g;

  //! @brief compensator of kernel intensity of node j on node i between t_i_k and t_i_(k-1)
  ArrayDouble2dList1D G;

  //! @brief compensator of kernel intensity of node j on node i between 0 and end_time
  ArrayDoubleList1D sum_G;

 public:
  /**
   * @brief Constructor
   * \param n_threads : number of threads that will be used for parallel computations
   */
  explicit ModelHawkesLogLikSingle(const int max_n_threads = 1);

  /**
   * @brief Precomputations of intermediate values
   * They will be used to compute faster loss, gradient and hessian norm.
   * \note This computation will be needed again if user modifies decay afterwards.
   */
  void compute_weights();

  /**
   * @brief Compute loss and gradient
   * \param coeffs : Point in which loss and gradient are computed
   * \param out : Array in which the value of the gradient is stored
   * \return Loss' value
   */
  double loss_and_grad(const ArrayDouble &coeffs, ArrayDouble &out);

  /**
   * @brief Compute loss
   * \param coeffs : Point in which loss is computed
   * \return Loss' value
   */
  double loss(const ArrayDouble &coeffs) override;

  /**
   * @brief Compute loss corresponding to sample i (between 0 and rand_max)
   * \param i : selected sample
   * \param coeffs : Point in which loss is computed
   * \return Loss' value
   * \note The sample i corresponds to the ith timestamp when looking component per component,
   * each component being sorted in temporal order
   */
  double loss_i(const ulong i, const ArrayDouble &coeffs) override;

  /**
   * @brief Compute gradient
   * \param coeffs : Point in which gradient is computed
   * \param out : Array in which the value of the gradient is stored
   */
  void grad(const ArrayDouble &coeffs, ArrayDouble &out) override;

  /**
   * @brief Compute gradient corresponding to sample i (between 0 and rand_max)
   * \param i : selected sample
   * \param coeffs : Point in which gradient is computed
   * \param out : Array in which the value of the gradient is stored
   * \note The sample i corresponds to the ith timestamp when looking component per component,
   * each component being sorted in temporal order
   */
  void grad_i(const ulong i, const ArrayDouble &coeffs, ArrayDouble &out) override;

  /**
   * @brief Compute the hessian norm \f$ \sqrt{ d^T \nabla^2 f(x) d} \f$
   * \param coeffs : Point in which the hessian is computed (\f$ x \f$)
   * \param vector : Point of which the norm is computed (\f$ d \f$)
   */
  double hessian_norm(const ArrayDouble &coeffs, const ArrayDouble &vector);

  /**
   * @brief Compute hessian
   * \param coeffs : Point in which hessian is computed
   * \param out : Array in which the value of the hessian is stored
   * \note : We only fill data, python code takes care of creating index and indexptr
   */
  void hessian(const ArrayDouble &coeffs, ArrayDouble &out);

 protected:
  virtual void allocate_weights();
  /**
   * @brief Precomputations of intermediate values for component i
   * \param i : selected component
   */
  virtual void compute_weights_dim_i(const ulong i);

  /**
   * @brief Convert sample i (between 0 and rand_max) to a tuple component, timestamp index
   * \param samples_d : selected sample
   * \param i : Where the component will be stored
   * \param k : Where the timestamp index will be stored
   */
  void sampled_i_to_index(const ulong sampled_i, ulong *i, ulong *k);

  /**
   * @brief Compute loss corresponding to component i
   * \param i : selected component
   * \param coeffs : Point in which loss is computed
   * \param out : Array which the result of the gradient will be added to
   * \return Loss' value
   * \note For two different values of i, this function will modify different coordinates of
   * out. Hence, it is thread safe.
   */
  double loss_dim_i(const ulong i, const ArrayDouble &coeffs);

  /**
   * @brief Compute loss corresponding to timestamp k of component i
   * \param i : selected component
   * \param k : selected timestamp index
   * \param coeffs : Point in which loss is computed
   * \return Loss' value
   */
  double loss_i_k(const ulong i, const ulong k, const ArrayDouble &coeffs);

  /**
   * @brief Compute gradient corresponding to component i
   * \param i : selected component
   * \param coeffs : Point in which gradient is computed
   * \param out : Array which the result of the gradient will be added to
   * \note For two different values of i, this function will modify different coordinates of
   * out. Hence, it is thread safe.
   */
  void grad_dim_i(const ulong i, const ArrayDouble &coeffs, ArrayDouble &out);

  /**
   * @brief Compute gradient corresponding to timestamp k of component i
   * \param i : selected component
   * \param k : selected timestamp index
   * \param coeffs : Point in which gradient is computed
   * \param out : Array which the result of the gradient will be added to
   */
  void grad_i_k(const ulong i, const ulong k, const ArrayDouble &coeffs, ArrayDouble &out);

  /**
   * @brief Compute loss and gradient
   * \param i : selected component
   * \param coeffs : Point in which loss and gradient are computed
   * \param out : Array which the result of the gradient will be added to
   * \note For two different values of i, this function will modify different coordinates of
   * out. Hence, it is thread safe.
   */
  double loss_and_grad_dim_i(const ulong i, const ArrayDouble &coeffs, ArrayDouble &out);

  /**
   * @brief Compute the hessian norm \f$ \sqrt{ d^T \nabla^2 f(x) d} \f$
   * \param i : selected component
   * \param coeffs : Point in which the hessian is computed (\f$ x \f$)
   * \param vector : Point of which the norm is computed (\f$ d \f$)
   */
  double hessian_norm_dim_i(const ulong i, const ArrayDouble &coeffs, const ArrayDouble &vector);

  /**
   * @brief Compute hessian corresponding to sample i (between 0 and rand_max = n_total_jumps)
   * \param i : selected dimension
   * \param coeffs : Point in which hessian is computed
   * \param out : Array in which the value of the hessian is stored
   * \note : We only fill data, python code takes care of creating index and indexptr
   */
  void hessian_i(const ulong i, const ArrayDouble &coeffs, ArrayDouble &out);

  /**
   * @brief Return the start of alpha i coefficients in a coeffs vector
   * @param i : selected dimension
   */
  virtual ulong get_alpha_i_first_index(const ulong i) const {
    TICK_CLASS_DOES_NOT_IMPLEMENT("");
  }

  /**
   * @brief Return the end of alpha i coefficients in a coeffs vector
   * @param i : selected dimension
   */
  virtual ulong get_alpha_i_last_index(const ulong i) const {
    TICK_CLASS_DOES_NOT_IMPLEMENT("");
  }

 public:
  //! @brief Returns max of the range of feasible grad_i and loss_i (total number of timestamps)
  inline ulong get_rand_max() const {
    return n_total_jumps;
  }

  friend ModelHawkesLogLik;
};

#endif  // LIB_INCLUDE_TICK_HAWKES_MODEL_BASE_MODEL_HAWKES_LOGLIK_SINGLE_H_
