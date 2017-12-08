#ifndef TICK_OPTIM_MODEL_SRC_VARIANTS_HAWKES_FIXED_EXPKERN_LEASTSQ_LIST_H_
#define TICK_OPTIM_MODEL_SRC_VARIANTS_HAWKES_FIXED_EXPKERN_LEASTSQ_LIST_H_

// License: BSD 3 clause

#include "tick/base/base.h"
#include "hawkes_leastsq_list.h"
#include "tick/hawkes/model/hawkes_fixed_expkern_leastsq.h"

/** \class ModelHawkesFixedExpKernLeastSqList
 * \brief Class for computing L2 Contrast function and gradient for Hawkes processes with
 * exponential kernels with fixed exponent (i.e., alpha*beta*e^{-beta t}, with fixed beta)
 * on a list of realizations
 */
class DLL_PUBLIC ModelHawkesFixedExpKernLeastSqList : public ModelHawkesLeastSqList {
  //! @brief Some arrays used for intermediate computings. They are initialized in init()
  ArrayDouble2d E, Dg, Dg2, C;

  //! @brief The 2d array of decays (remember that the decays are fixed!)
  SArrayDouble2dPtr decays;

 public:
  //! @brief Empty constructor
  //! This constructor should only be used for serialization
  ModelHawkesFixedExpKernLeastSqList(): ModelHawkesLeastSqList(0, 0) {}

  //! @brief Constructor
  //! \param decays : the 2d array of the decays
  //! \param max_n_threads : number of cores to be used for multithreading. If negative,
  //! the number of physical cores will be used
  //! \param optimization_level : 0 corresponds to no optimization and 1 to use of faster (approximated) exponential function
  ModelHawkesFixedExpKernLeastSqList(const SArrayDouble2dPtr decays,
                                     const int max_n_threads = 1,
                                     const unsigned int optimization_level = 0);

  /**
   * @brief Compute hessian
   * \param coeffs : Point in which hessian is computed
   * \param out : Array in which the value of the hessian is stored
   * \note : We only fill data, python code takes care of creating index and indexptr
   */
  void hessian(ArrayDouble &out);

  /**
   * @brief Set decays and reset weights computing
   * @param decays : new decays to be set
   */
  void set_decays(const SArrayDouble2dPtr decays) {
    weights_computed = false;
    if (decays->n_rows() != n_nodes || decays->n_cols() != n_nodes) {
      TICK_ERROR("decays must be (" << n_nodes << ", " << n_nodes << ") array"
                                    << " but recevied a (" << decays->n_rows() << ", "
                                    << decays->n_cols() << ") array");
    }
    this->decays = decays;
  }

  ulong get_n_coeffs() const override;

 private:
  /**
   * @brief Compute weights for one index between 0 and n_realizations * n_nodes
   * @param i_r : r * n_realizations + i, tells which realization and which node
   * @param model_list : list of models on which to compute and store weights. Only model_list[r]
   * will be modified
   */
  void compute_weights_i_r(const ulong i_r,
                           std::vector<ModelHawkesFixedExpKernLeastSq> &model_list);

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
    ar(cereal::make_nvp("ModelHawkesLeastSqList", cereal::base_class<ModelHawkesLeastSqList>(this)));

    ar(CEREAL_NVP(E));
    ar(CEREAL_NVP(Dg));
    ar(CEREAL_NVP(Dg2));
    ar(CEREAL_NVP(C));
    ar(CEREAL_NVP(decays));
  }
};

#endif  // TICK_OPTIM_MODEL_SRC_VARIANTS_HAWKES_FIXED_EXPKERN_LEASTSQ_LIST_H_
