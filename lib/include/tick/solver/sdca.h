//
// Created by Martin Bompaire on 22/10/15.
//

#ifndef LIB_INCLUDE_TICK_SOLVER_SDCA_H_
#define LIB_INCLUDE_TICK_SOLVER_SDCA_H_

// License: BSD 3 clause

#include "sto_solver.h"
#include "tick/base_model/model.h"
#include "tick/base_model/model_generalized_linear.h"

// TODO: profile the code of SDCA to check if it's faster
// TODO: code accelerated SDCA

template <class T, class K>
class DLL_PUBLIC TBaseSDCA : public TStoSolver<T, K> {
 protected:
  using TStoSolver<T, K>::t;
  using TStoSolver<T, K>::model;
  using TStoSolver<T, K>::iterate;
  using TStoSolver<T, K>::prox;
  using TStoSolver<T, K>::get_next_i;
  using TStoSolver<T, K>::epoch_size;
  using TStoSolver<T, K>::rand_max;

 public:
  using TStoSolver<T, K>::get_class_name;
  using SArrayTPtr = std::shared_ptr<SArray<T>>;

 protected:
  ulong n_coeffs;

  // A boolean that attests that our arrays of ascent variables and dual
  // variables are initialized with the right size
  bool stored_variables_ready;

  // Store for coefficient update before prox call.
  Array<K> tmp_primal_vector;

  // Level of ridge regularization. This is mandatory for SDCA.
  T l_l2sq;

  // Ascent variables
  Array<T> delta;

  // The dual variable
  Array<K> dual_vector;

  std::shared_ptr<TModelGeneralizedLinear<T, K>> casted_model;
  std::shared_ptr<TProxSeparable<T, K>> casted_prox;

  bool ready_step_corrections = false;
  // Probabilistic correction of the step-sizes of all model weights,
  // given by the inverse proportion of non-zero entries in each feature column
  Array<T> steps_correction;

 public:
  // This exists soley for cereal/swig
  TBaseSDCA() : TBaseSDCA<T, K>(0, 0, 0) {}

  explicit TBaseSDCA(T l_l2sq, ulong epoch_size = 0, T tol = 0.,
                 RandType rand_type = RandType::unif,  int record_every = 1, int seed = -1);

  void reset() override;

  void set_model(std::shared_ptr<TModel<T, K>> model) override;

  T get_l_l2sq() const { return l_l2sq; }

  void set_l_l2sq(T l_l2sq) { this->l_l2sq = l_l2sq; }

  SArrayTPtr get_primal_vector() const {
    Array<T> copy(iterate.size());
    copy.init_to_zero();
    copy.mult_incr(iterate, 1);
    return copy.as_sarray_ptr();
  }

  SArrayTPtr get_dual_vector() const {
    Array<T> copy(dual_vector.size());
    copy.init_to_zero();
    copy.mult_incr(dual_vector, 1);
    return copy.as_sarray_ptr();
  }

  void set_starting_iterate();
  void set_starting_iterate(Array<K> &dual_vector) override;

 protected:
  T get_scaled_l_l2sq() const {
    // In order to solve the same problem than other solvers, we need to rescale
    // the penalty parameter if some observations are not considered in SDCA.
    // This is useful for Poisson regression with identity link
    return l_l2sq * model->get_n_samples() / rand_max;
  }

  void compute_step_corrections();

// public:
//  template <class Archive>
//  void serialize(Archive &ar) {
//    ar(cereal::make_nvp("StoSolver",
//                        cereal::base_class<TStoSolver<T, K>>(this)));
//
//    ar(CEREAL_NVP(n_coeffs));
//    ar(CEREAL_NVP(stored_variables_ready));
//    ar(CEREAL_NVP(l_l2sq));
//    ar(CEREAL_NVP(delta));
//    ar(CEREAL_NVP(dual_vector));
//  }
//
//  BoolStrReport compare(const TBaseSDCA<T, K> &that) {
//    std::stringstream ss;
//    ss << get_class_name() << std::endl;
//    bool are_equal =
//        TStoSolver<T, K>::compare(that, ss) && TICK_CMP_REPORT(ss, n_coeffs) &&
//        TICK_CMP_REPORT(ss, stored_variables_ready) &&
//        TICK_CMP_REPORT(ss, l_l2sq) && TICK_CMP_REPORT(ss, delta) &&
//        TICK_CMP_REPORT(ss, dual_vector);
//    return BoolStrReport(are_equal, ss.str());
//  }
//
//  BoolStrReport operator==(const TBaseSDCA<T, K> &that) { return compare(that); }
//
//  static std::shared_ptr<TBaseSDCA<T, K>> AS_NULL() {
//    return std::move(std::shared_ptr<TBaseSDCA<T, K>>(new TBaseSDCA<T, K>));
//  }
};

template <class T>
class DLL_PUBLIC TSDCA : public TBaseSDCA<T, T> {
 public:
  using TBaseSDCA<T, T>::t;
  using TBaseSDCA<T, T>::model;
  using TBaseSDCA<T, T>::iterate;
  using TBaseSDCA<T, T>::prox;
  using TBaseSDCA<T, T>::get_next_i;
  using TBaseSDCA<T, T>::epoch_size;
  using TBaseSDCA<T, T>::rand_max;
  using TBaseSDCA<T, T>::stored_variables_ready;
  using TBaseSDCA<T, T>::get_scaled_l_l2sq;
  using TBaseSDCA<T, T>::set_starting_iterate;
  using TBaseSDCA<T, T>::delta;
  using TBaseSDCA<T, T>::dual_vector;

 public:
  TSDCA() : TSDCA<T>(0, 0, 0) {}

  explicit TSDCA(T l_l2sq, ulong epoch_size = 0, T tol = 0.,
      RandType rand_type = RandType::unif,  int record_every = 1, int seed = -1);

  void solve_one_epoch() override;
};

using SDCA = TSDCA<double>;

using SDCADouble = TSDCA<double>;
//CEREAL_SPECIALIZE_FOR_ALL_ARCHIVES(SDCADouble,
//                                   cereal::specialization::member_serialize)
//CEREAL_REGISTER_TYPE(SDCADouble)

using SDCAFloat = TSDCA<float>;
//CEREAL_SPECIALIZE_FOR_ALL_ARCHIVES(SDCAFloat,
//                                   cereal::specialization::member_serialize)
//CEREAL_REGISTER_TYPE(SDCAFloat)

#endif  // LIB_INCLUDE_TICK_SOLVER_SDCA_H_
