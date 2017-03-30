//
// Created by Martin Bompaire on 16/07/15.
//

#ifndef TICK_OPTIM_MODEL_SRC_HAWKES_FIXED_EXPKERN_LOGLIK_H_
#define TICK_OPTIM_MODEL_SRC_HAWKES_FIXED_EXPKERN_LOGLIK_H_

#include "base.h"

#include "base/hawkes_single.h"

class ModelHawkesFixedExpKernLogLikList;

/**
 * \class ModelHawkesFixedExpKernLogLik
 * \brief Class for computing loglikelihood function and gradient for Hawkes processes with
 * exponential kernels with fixed exponent (i.e., \f$ \alpha \beta e^{-\beta t} \f$, with fixed
 * decay)
 */
class ModelHawkesFixedExpKernLogLik : public ModelHawkesSingle {
 private:
  //! @brief Value of decay for this model
  double decay;

  //! @brief Some arrays used for intermediate computings. They are initialized in init()
  ArrayDouble2dList1D g;
  ArrayDouble2dList1D G;
  ArrayDoubleList1D sum_G;

 public:
  //! @brief Default constructor
  //! @note This constructor is only used to create vectors of ModelHawkesFixedExpKernLeastSq
  ModelHawkesFixedExpKernLogLik() : ModelHawkesSingle() {}

  /**
   * @brief Constructor
   * \param decay : decay for this model (remember that decay is fixed!)
   * \param n_threads : number of threads that will be used for parallel computations
   */
  explicit ModelHawkesFixedExpKernLogLik(const double decay, const int max_n_threads = 1);

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

 private:
  void allocate_weights();
  /**
   * @brief Precomputations of intermediate values for component i
   * \param i : selected component
   */
  void compute_weights_dim_i(const ulong i);

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

 public:
  ulong get_n_coeffs() const override;

  //! @brief Returns max of the range of feasible grad_i and loss_i (total number of timestamps)
  inline ulong get_rand_max() const {
    return n_total_jumps;
  }
  //! @brief Returns decay that was set
  double get_decay() const {
    return decay;
  }

  /**
   * @brief Set new decay
   * \param decay : new decay
   * \note Weights will need to be recomputed
   */
  void set_decay(double decay) {
    this->decay = decay;
    weights_computed = false;
  }

  friend ModelHawkesFixedExpKernLogLikList;
};

#endif  // TICK_OPTIM_MODEL_SRC_HAWKES_FIXED_EXPKERN_LOGLIK_H_
