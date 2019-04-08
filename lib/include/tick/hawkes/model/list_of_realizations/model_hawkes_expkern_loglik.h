
#ifndef LIB_INCLUDE_TICK_HAWKES_MODEL_LIST_OF_REALIZATIONS_MODEL_HAWKES_EXPKERN_LOGLIK_H_
#define LIB_INCLUDE_TICK_HAWKES_MODEL_LIST_OF_REALIZATIONS_MODEL_HAWKES_EXPKERN_LOGLIK_H_

// License: BSD 3 clause

#include "tick/base/base.h"
#include "tick/hawkes/model/base/model_hawkes_loglik.h"
#include "tick/hawkes/model/model_hawkes_expkern_loglik_single.h"

/** \class ModelHawkesExpKernLogLik
 * \brief Class for computing L2 Contrast function and gradient for Hawkes
 * processes with exponential kernels with fixed exponent (i.e.,
 * alpha*beta*e^{-beta t}, with fixed beta) on a list of realizations
 */
class DLL_PUBLIC ModelHawkesExpKernLogLik : public ModelHawkesLogLik {
  //! @brief Value of decay for this model. Shared by all kernels
  double decay;

 public:
  ModelHawkesExpKernLogLik() {}
  /**
   * @brief Constructor
   * \param decay : decay for this model (remember that decay is fixed!)
   * \param max_n_threads : number of cores to be used for multithreading. If
   * negative, the number of physical cores will be used
   */
  explicit ModelHawkesExpKernLogLik(const double decay,
                                    const int max_n_threads = 1);

  /**
   * @brief Set decays and reset weights computing
   * @param decays : new decays to be set
   */
  void set_decay(const double decay) {
    weights_computed = false;
    this->decay = decay;
  }

  double get_decay() const { return decay; }

  std::unique_ptr<ModelHawkesLogLikSingle> build_model(
      const int n_threads) override {
    return std::unique_ptr<ModelHawkesExpKernLogLikSingle>(
        new ModelHawkesExpKernLogLikSingle(decay, n_threads));
  }

  ulong get_n_coeffs() const override;

  template <class Archive>
  void serialize(Archive &ar) {
    ar(cereal::make_nvp("ModelHawkesLogLik",
                        cereal::base_class<ModelHawkesLogLik>(this)));

    ar(CEREAL_NVP(decay));
  }

  BoolStrReport compare(const ModelHawkesExpKernLogLik &that, std::stringstream &ss) {
    ss << get_class_name() << std::endl;
    auto are_equal = ModelHawkesLogLik::compare(that, ss) &&
                     TICK_CMP_REPORT(ss, decay);
    return BoolStrReport(are_equal, ss.str());
  }
  BoolStrReport compare(const ModelHawkesExpKernLogLik &that) {
    std::stringstream ss;
    return compare(that, ss);
  }
  BoolStrReport operator==(const ModelHawkesExpKernLogLik &that) {
    return ModelHawkesExpKernLogLik::compare(that);
  }
};

CEREAL_SPECIALIZE_FOR_ALL_ARCHIVES(ModelHawkesExpKernLogLik,
                                   cereal::specialization::member_serialize)
CEREAL_REGISTER_TYPE(ModelHawkesExpKernLogLik)

#endif  // LIB_INCLUDE_TICK_HAWKES_MODEL_LIST_OF_REALIZATIONS_MODEL_HAWKES_EXPKERN_LOGLIK_H_
