#ifndef TICK_OPTIM_MODEL_SRC_VARIANTS_HAWKES_FIXED_EXPKERN_LOGLIK_LIST_H_
#define TICK_OPTIM_MODEL_SRC_VARIANTS_HAWKES_FIXED_EXPKERN_LOGLIK_LIST_H_

#include "base.h"
#include "../base/hawkes_list.h"
#include "../hawkes_fixed_expkern_loglik.h"

/** \class ModelHawkesFixedExpKernLogLikList
 * \brief Class for computing L2 Contrast function and gradient for Hawkes processes with
 * exponential kernels with fixed exponent (i.e., alpha*beta*e^{-beta t}, with fixed beta)
 * on a list of realizations
 */
class ModelHawkesFixedExpKernLogLikList : public ModelHawkesList {
  //! @brief Value of decay for this model. Shared by all kernels
  double decay;

  std::vector<ModelHawkesFixedExpKernLogLik> model_list;

 public:
  /**
   * @brief Constructor
   * \param decay : decay for this model (remember that decay is fixed!) 
   * \param max_n_threads : number of cores to be used for multithreading. If negative,
   * the number of physical cores will be used
   */
  ModelHawkesFixedExpKernLogLikList(const double decay,
                                    const int max_n_threads = 1);

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

  /**
   * @brief Compute loss and gradient
   * \param coeffs : Point in which loss and gradient are computed
   * \param out : Array in which the value of the gradient is stored
   * \return Loss' value
   */
  double loss_and_grad(const ArrayDouble &coeffs, ArrayDouble &out);

  /**
   * @brief Compute the hessian norm \f$ \sqrt{ d^T \nabla^2 f(x) d} \f$
   * \param coeffs : Point in which the hessian is computed (\f$ x \f$)
   * \param vector : Point of which the norm is computed (\f$ d \f$)
   */
  double hessian_norm(const ArrayDouble &coeffs, const ArrayDouble &vector);

  /**
   * @brief Set decays and reset weights computing
   * @param decays : new decays to be set
   */
  void set_decay(const double decay) {
    weights_computed = false;
    this->decay = decay;
  }

  ulong get_rand_max() const {
    return get_n_total_jumps();
  }

  ulong get_n_coeffs() const override;

 private:
  /**
   * @brief Converts index between 0 and n_realizations * n_nodes to corresponding
   * realization and node
   * @param i_r : original index
   * @return tuple containing first the realization, then the node
   */
  std::tuple<ulong, ulong> get_realization_node(ulong i_r);

  /**
   * @brief Compute weights for one index between 0 and n_realizations * n_nodes
   * @param i_r : r * n_realizations + i, tells which realization and which node
   */
  void compute_weights_i_r(const ulong i_r);

  /**
   * @brief Compute loss for one index between 0 and n_realizations * n_nodes
   * @param i_r : r * n_realizations + i, tells which realization and which node
   * \param coeffs : Point in which loss is computed
   */
  double loss_i_r(const ulong i_r, const ArrayDouble &coeffs);

  /**
   * @brief Compute gradient for one index between 0 and n_realizations * n_nodes
   * \param i_r : r * n_realizations + i, tells which realization and which node
   * \param coeffs : Point in which gradient is computed
   * \param out : Array in which the value of the gradient is stored
   */
  void grad_i_r(const ulong i_r, ArrayDouble &out, const ArrayDouble &coeffs);

  /**
   * @brief Compute the hessian norm for one index between 0 and n_realizations * n_nodes
   * \param i_r : r * n_realizations + i, tells which realization and which node
   * \param coeffs : Point in which the hessian is computed
   * \param vector : Point of which the norm is computed
   */
  double hessian_norm_i_r(const ulong i_r, const ArrayDouble &coeffs,
                          const ArrayDouble &vector);

  std::pair<ulong, ulong> sampled_i_to_realization(const ulong sampled_i);
};

#endif  // TICK_OPTIM_MODEL_SRC_VARIANTS_HAWKES_FIXED_EXPKERN_LOGLIK_LIST_H_
