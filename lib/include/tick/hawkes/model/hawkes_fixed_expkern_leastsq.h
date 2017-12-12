#ifndef TICK_OPTIM_MODEL_SRC_HAWKES_FIXED_EXPKERN_LEASTSQ_H_
#define TICK_OPTIM_MODEL_SRC_HAWKES_FIXED_EXPKERN_LEASTSQ_H_

// License: BSD 3 clause

#include "tick/base/base.h"
#include "base/hawkes_single.h"

class ModelHawkesFixedExpKernLeastSqList;

/** \class ModelHawkesFixedExpKernLeastSq
 * \brief Class for computing L2 Contrast function and gradient for Hawkes processes with
 * exponential kernels with fixed exponent (i.e., alpha*beta*e^{-beta t}, with fixed beta)
 */
class DLL_PUBLIC ModelHawkesFixedExpKernLeastSq : public ModelHawkesSingle {
  //! @brief Some arrays used for intermediate computings. They are initialized in init()
  ArrayDouble2d E, Dg, Dg2, C;

  //! @brief The 2d array of decays (remember that the decays are fixed!)
  SArrayDouble2dPtr decays;

 public:
  //! @brief Default constructor
  //! @note This constructor is only used to create vectors of ModelHawkesFixedExpKernLeastSq
  ModelHawkesFixedExpKernLeastSq() : ModelHawkesSingle() {}

  //! @brief Constructor
  //! \param decays : the 2d array of the decays
  //! \param max_n_threads : maximum number of threads to be used for multithreading
  //! \param optimization_level : 0 corresponds to no optimization and 1 to use of faster (approximated) exponential function
  ModelHawkesFixedExpKernLeastSq(const SArrayDouble2dPtr decays,
                                 const int max_n_threads = 1,
                                 const unsigned int optimization_level = 0);

  /**
   * @brief Precomputations of intermediate values
   * They will be used to compute faster loss, gradient and hessian norm.
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
   * @brief Compute hessian
   * \param coeffs : Point in which hessian is computed
   * \param out : Array in which the value of the hessian is stored
   * \note : We only fill data, python code takes care of creating index and indexptr
   */
  void hessian(ArrayDouble &out);

  /**
   * @brief Compute loss and gradient
   * \param coeffs : Point in which loss and gradient are computed
   * \param out : Array in which the value of the gradient is stored
   * \return Loss' value
   */
  double loss_and_grad(const ArrayDouble &coeffs, ArrayDouble &out);

  void set_decays(const SArrayDouble2dPtr decays) {
    this->decays = decays;
    weights_computed = false;
  }

  ulong get_n_coeffs() const override;

 private:
  void allocate_weights();
  /**
   * @brief Precomputations of intermediate values for dimension i
   * \param i : selected dimension
   */
  void compute_weights_i(const ulong i);

  /**
   * @brief Compute hessian corresponding to sample i (between 0 and rand_max = dim)
   * \param i : selected dimension
   * \param coeffs : Point in which hessian is computed
   * \param out : Array in which the value of the hessian is stored
   * \note : We only fill data, python code takes care of creating index and indexptr
   */
  void hessian_i(const ulong i, ArrayDouble &out);

  friend class ModelHawkesFixedExpKernLeastSqList;

 public:
  template<class Archive>
  void serialize(Archive &ar) {
    ar(cereal::make_nvp("ModelHawkesSingle", cereal::base_class<ModelHawkesSingle>(this)));

    ar(CEREAL_NVP(E));
    ar(CEREAL_NVP(Dg));
    ar(CEREAL_NVP(Dg2));
    ar(CEREAL_NVP(C));
    ar(CEREAL_NVP(decays));
  }
};

CEREAL_REGISTER_TYPE(ModelHawkesFixedExpKernLeastSq);

#endif  // TICK_OPTIM_MODEL_SRC_HAWKES_FIXED_EXPKERN_LEASTSQ_H_
