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
  using TStoSolver<T, K>::get_generator;

  using TStoSolver<T, K>::save_history;
  using TStoSolver<T, K>::last_record_epoch;
  using TStoSolver<T, K>::last_record_time;
  using TStoSolver<T, K>::record_every;
  using TStoSolver<T, K>::rand;

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

  size_t n_threads = 0;

  T step_size = 1.;


 public:
  // This exists soley for cereal/swig
  TBaseSDCA() : TBaseSDCA<T, K>(0, 0, 0) {}

  explicit TBaseSDCA(T l_l2sq, ulong epoch_size, T tol = 0.,
                     RandType rand_type = RandType::unif,  size_t record_every = 1, int seed = -1,
                     int n_threads = 1);

  void reset() override;

  void set_model(std::shared_ptr<TModel<T, K>> model) override;

  T get_l_l2sq() const { return l_l2sq; }

  void set_l_l2sq(T l_l2sq) { this->l_l2sq = l_l2sq; }

  void set_step_size(T step_size) { this->step_size = step_size; }

  SArrayTPtr get_primal_vector() const {
    // This works when K is atomic
    Array<T> copy(iterate.size());
    copy.init_to_zero();
    copy.mult_incr(iterate, 1);
    return copy.as_sarray_ptr();
  }

  SArrayTPtr get_dual_vector() const {
    // This works when K is atomic
    Array<T> copy(dual_vector.size());
    copy.init_to_zero();
    copy.mult_incr(dual_vector, 1);
    return copy.as_sarray_ptr();
  }

  void prepare_solve();
  void solve(size_t n_epochs = 1) override;
  void solve_batch(size_t n_epochs = 1, ulong batch_size = 1);
  void set_starting_iterate();
  void set_starting_iterate(Array<T> &dual_vector) override;

 protected:
  void precompute_sdca_dual_min_weights(
      Array<K> &local_iterate, ulong batch_size, double _1_over_lbda_n,
      const ArrayULong &feature_indices, Array2d<T> &g, Array<T> &p);

  virtual void update_delta_dual_i(ulong i, double delta_dual_i,
                                   const BaseArray<T> &feature_i, double _1_over_lbda_n);

  virtual void update_delta_dual_batch(ArrayULong &indices, ArrayULong &feature_indices,
                                       Array<T> &delta_duals, double _1_over_lbda_n);

  void tmp_iterate_to_iterate(Array<K> &iterate);

 public:
  template <class Archive>
  void serialize(Archive &ar) {
    ar(cereal::make_nvp("StoSolver",
                       cereal::base_class<TStoSolver<T, K>>(this)));

    ar(CEREAL_NVP(n_coeffs));
    ar(CEREAL_NVP(stored_variables_ready));
    ar(CEREAL_NVP(l_l2sq));
    ar(CEREAL_NVP(delta));
    ar(CEREAL_NVP(dual_vector));
  }

  BoolStrReport compare(const TBaseSDCA<T, K> &that) {
    std::stringstream ss;
    ss << get_class_name() << std::endl;
    bool are_equal =
        TStoSolver<T, K>::compare(that, ss) && TICK_CMP_REPORT(ss, n_coeffs) &&
        TICK_CMP_REPORT(ss, stored_variables_ready) &&
        TICK_CMP_REPORT(ss, l_l2sq) && TICK_CMP_REPORT(ss, delta) &&
        TICK_CMP_REPORT(ss, dual_vector);
    return BoolStrReport(are_equal, ss.str());
  }

  BoolStrReport operator==(const TBaseSDCA<T, K> &that) { return compare(that); }
};

template <class T>
class DLL_PUBLIC TSDCA : public TBaseSDCA<T, T> {
 public:
  using TBaseSDCA<T, T>::model;
  using TBaseSDCA<T, T>::delta;
  using TBaseSDCA<T, T>::dual_vector;
  using TBaseSDCA<T, T>::tmp_primal_vector;

 public:
  TSDCA() : TSDCA<T>(0, 0, 0) {}

  explicit TSDCA(T l_l2sq, ulong epoch_size, T tol = 0.,
      RandType rand_type = RandType::unif,  size_t record_every = 1, int seed = -1,
      int n_threads = 1);

 protected:
  void update_delta_dual_i(ulong i, double delta_dual_i,
                           const BaseArray<T> &feature_i, double _1_over_lbda_n) override;

  void update_delta_dual_batch(ArrayULong &indices, ArrayULong &feature_indices,
                               Array<T> &delta_duals, double _1_over_lbda_n) override;
};

using SDCADouble = TSDCA<double>;
CEREAL_SPECIALIZE_FOR_ALL_ARCHIVES(SDCADouble,
                                  cereal::specialization::member_serialize)
CEREAL_REGISTER_TYPE(SDCADouble)

using SDCAFloat = TSDCA<float>;
CEREAL_SPECIALIZE_FOR_ALL_ARCHIVES(SDCAFloat,
                                  cereal::specialization::member_serialize)
CEREAL_REGISTER_TYPE(SDCAFloat)

template <class T>
class DLL_PUBLIC TAtomicSDCA : public TBaseSDCA<T, std::atomic<T> > {
 protected:
  using TBaseSDCA<T, std::atomic<T>>::model;
  using TBaseSDCA<T, std::atomic<T>>::delta;
  using TBaseSDCA<T, std::atomic<T>>::dual_vector;
  using TBaseSDCA<T, std::atomic<T>>::tmp_primal_vector;

 public:
  // This exists solely for cereal/swig
  TAtomicSDCA() : TAtomicSDCA<T>(0, 0, 0) {}

  explicit TAtomicSDCA(T l_l2sq, ulong epoch_size, T tol = 0.,
                       RandType rand_type = RandType::unif, size_t record_every = 1, int seed = -1,
                       int n_threads = 2);

 protected:
  void update_delta_dual_i(ulong i, double delta_dual_i,
                           const BaseArray<T> &feature_i, double _1_over_lbda_n) override;

  void update_delta_dual_batch(ArrayULong &indices, ArrayULong &feature_indices,
                               Array<T> &delta_duals, double _1_over_lbda_n) override;
};

using AtomicSDCA = TAtomicSDCA<double>;

using AtomicSDCADouble = TAtomicSDCA<double>;
// CEREAL_SPECIALIZE_FOR_ALL_ARCHIVES(SDCADouble,
//                                   cereal::specialization::member_serialize)
// CEREAL_REGISTER_TYPE(SDCADouble)

using AtomicSDCAFloat = TAtomicSDCA<float>;
// CEREAL_SPECIALIZE_FOR_ALL_ARCHIVES(SDCAFloat,
//                                   cereal::specialization::member_serialize)
// CEREAL_REGISTER_TYPE(SDCAFloat)

#endif  // LIB_INCLUDE_TICK_SOLVER_SDCA_H_
