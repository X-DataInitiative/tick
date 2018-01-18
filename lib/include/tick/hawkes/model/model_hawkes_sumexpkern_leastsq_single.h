
#ifndef LIB_INCLUDE_TICK_HAWKES_MODEL_MODEL_HAWKES_SUMEXPKERN_LEASTSQ_SINGLE_H_
#define LIB_INCLUDE_TICK_HAWKES_MODEL_MODEL_HAWKES_SUMEXPKERN_LEASTSQ_SINGLE_H_

// License: BSD 3 clause

#include "tick/base/base.h"
#include "tick/hawkes/model/base/model_hawkes_single.h"

/** \class ModelHawkesSumExpKernLeastSqSingle
 * \brief Class for computing L2 Contrast function and gradient for Hawkes processes with
 * sum exponential kernels with fixed exponent (i.e., \sum_u alpha_u*beta_u*e^{-beta_u t},
 * with fixed beta)
 */
class DLL_PUBLIC ModelHawkesSumExpKernLeastSqSingle : public ModelHawkesSingle {
  //! @brief Some arrays used for intermediate computings.
  ArrayDouble2dList1D E, Dgg, C;

  //! @brief some arrays used for intermediate computings in varying baseline case
  ArrayDouble L;
  ArrayDoubleList1D K;
  ArrayDouble2dList1D Dg;

  ulong n_baselines;
  double period_length;

  //! @brief The array of decays (remember that the decays are fixed!)
  ArrayDouble decays;

  //! @brief n_decays (number of decays in the sum exponential kernel)
  ulong n_decays;

 public:
  //! @brief Default constructor
  //! @note This constructor is only used to create vectors of ModelHawkesSumExpKernLeastSqSingle and serialization
  ModelHawkesSumExpKernLeastSqSingle() {}

  //! @brief Constructor
  //! \param timestamps : a list of arrays representing the realization
  //! \param decays : the 2d array of the decays
  //! \param end_time : The time until which this process has been observed
  //! \param max_n_threads : maximum number of threads to be used for multithreading
  //! \param optimization_level : 0 corresponds to no optimization and 1 to use of faster
  //! (approximated) exponential function
  ModelHawkesSumExpKernLeastSqSingle(const ArrayDouble &decays,
                                     const ulong n_baselines,
                                     const double period_length,
                                     const unsigned int max_n_threads = 1,
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
   * @brief Compute loss corresponding to sample i (between 0 and rand_max = n_nodes)
   * \param i : selected component
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
   * @brief Compute gradient corresponding to sample i (between 0 and rand_max = n_nodes)
   * \param i : selected component
   * \param coeffs : Point in which gradient is computed
   * \param out : Array in which the value of the gradient is stored
   */
  void grad_i(ulong i, const ArrayDouble &coeffs, ArrayDouble &out) override;

  /**
   * @brief Compute loss and gradient
   * \param coeffs : Point in which loss and gradient are computed
   * \param out : Array in which the value of the gradient is stored
   * \return Loss' value
   */
  double loss_and_grad(const ArrayDouble &coeffs, ArrayDouble &out);

  //! @brief Synchronize n_coeffs given other attributes
  ulong get_n_coeffs() const override;

  ulong get_n_baselines() const;
  double get_period_length() const;

  void set_n_baselines(ulong n_baselines);
  void set_period_length(double period_length);

 private:
  void allocate_weights();

  /**
   * @brief Precomputations of intermediate values for component i
   * \param i : selected component
   */
  void compute_weights_i(const ulong i);

  ulong get_baseline_interval(const double t);
  double get_baseline_interval_length(const ulong interval_p);

  friend class ModelHawkesSumExpKernLeastSq;

 public:
  template<class Archive>
  void serialize(Archive &ar) {
    ar(cereal::make_nvp("ModelHawkesSingle", cereal::base_class<ModelHawkesSingle>(this)));

    ar(CEREAL_NVP(E));
    ar(CEREAL_NVP(Dgg));
    ar(CEREAL_NVP(C));
    ar(CEREAL_NVP(L));
    ar(CEREAL_NVP(K));
    ar(CEREAL_NVP(Dg));
    ar(CEREAL_NVP(n_baselines));
    ar(CEREAL_NVP(period_length));
    ar(CEREAL_NVP(decays));
    ar(CEREAL_NVP(n_decays));
  }
};

CEREAL_REGISTER_TYPE(ModelHawkesSumExpKernLeastSqSingle);

#endif  // LIB_INCLUDE_TICK_HAWKES_MODEL_MODEL_HAWKES_SUMEXPKERN_LEASTSQ_SINGLE_H_
