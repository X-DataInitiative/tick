
#ifndef LIB_INCLUDE_TICK_HAWKES_MODEL_LIST_OF_REALIZATIONS_MODEL_HAWKES_SUMEXPKERN_LEASTSQ_H__
#define LIB_INCLUDE_TICK_HAWKES_MODEL_LIST_OF_REALIZATIONS_MODEL_HAWKES_SUMEXPKERN_LEASTSQ_H__

// License: BSD 3 clause

#include "tick/base/base.h"
#include "tick/base_model/model.h"
#include "tick/hawkes/model/model_hawkes_sumexpkern_leastsq_single.h"
#include "tick/hawkes/model/base/model_hawkes_leastsq.h"

/** \class ModelHawkesSumExpKernLeastSq
 * \brief Class for computing L2 Contrast function and gradient for Hawkes processes with
 * exponential kernels with fixed exponent (i.e., alpha*beta*e^{-beta t}, with fixed beta)
 */
class DLL_PUBLIC ModelHawkesSumExpKernLeastSq : public ModelHawkesLeastSq {
  //! @brief Some arrays used for intermediate computings.
  std::vector<ArrayDouble2d> E, Dgg, C;
  ArrayDouble2dList1D Dg;

  //! @brief some arrays used for intermediate computings in varying baseline case
  ArrayDouble L;
  ArrayDoubleList1D K;

  ulong n_baselines;
  double period_length;

  //! @brief The array of decays (remember that the decays are fixed!)
  ArrayDouble decays;

  //! @brief n_decays (number of decays in the sum exponential kernel)
  ulong n_decays;

 public:
  //! @brief Empty constructor
  //! This constructor should only be used for serialization
  ModelHawkesSumExpKernLeastSq() : ModelHawkesLeastSq(0, 0) {}

  //! @brief Constructor
  //! \param timestamps : a list of arrays representing the realization
  //! \param decays : the 2d array of the decays
  //! \param n_cores : number of cores to be used for multithreading
  //! \param optimization_level : 0 corresponds to no optimization and 1 to use of faster (approximated) exponential function
  ModelHawkesSumExpKernLeastSq(const ArrayDouble &decays,
                               const ulong n_baselines,
                               const double period_length,
                               const unsigned int max_n_threads = 1,
                               const unsigned int optimization_level = 0);

  void set_decays(const ArrayDouble &decays) {
    weights_computed = false;
    this->decays = decays;
    n_decays = decays.size();
  }

  //! @brief Synchronize n_coeffs given other attributes
  ulong get_n_coeffs() const override;

  ulong get_n_decays() const { return n_decays; }

  ulong get_n_baselines() const;
  double get_period_length() const;

  void set_n_baselines(ulong n_baselines);
  void set_period_length(double period_length);

 private:
  /**
   * @brief Compute weights for one index between 0 and n_realizations * n_nodes
   * @param i_r : r * n_realizations + i, tells which realization and which node
   * @param model_list : list of models on which to compute and store weights. Only model_list[r]
   * will be modified
   */
  void compute_weights_i_r(const ulong i_r,
                           std::vector<ModelHawkesSumExpKernLeastSqSingle> &model_list);

  //! @brief allocate arrays to store precomputations
  void allocate_weights() override;

  //! @brief synchronize aggregate_model with this instance
  void synchronize_aggregated_model() override;

  void compute_weights_timestamps_list() override;
  void compute_weights_timestamps(const SArrayDoublePtrList1D &timestamps,
                                  double end_time) override;

 public:
  template<class Archive>
  void serialize(Archive &ar) {
    ar(cereal::make_nvp("ModelHawkesLeastSq", cereal::base_class<ModelHawkesLeastSq>(this)));

    ar(CEREAL_NVP(E));
    ar(CEREAL_NVP(Dgg));
    ar(CEREAL_NVP(C));
    ar(CEREAL_NVP(Dg));
    ar(CEREAL_NVP(L));
    ar(CEREAL_NVP(K));
    ar(CEREAL_NVP(n_baselines));
    ar(CEREAL_NVP(period_length));
    ar(CEREAL_NVP(decays));
    ar(CEREAL_NVP(n_decays));
  }
};

#endif  // LIB_INCLUDE_TICK_HAWKES_MODEL_LIST_OF_REALIZATIONS_MODEL_HAWKES_SUMEXPKERN_LEASTSQ_H__
