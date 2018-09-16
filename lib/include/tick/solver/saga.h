#ifndef LIB_INCLUDE_TICK_SOLVER_SAGA_H_
#define LIB_INCLUDE_TICK_SOLVER_SAGA_H_

// License: BSD 3 clause

#include "sto_solver.h"
#include "tick/base_model/model_generalized_linear.h"

template <class T, class K>
class DLL_PUBLIC TBaseSAGA : public TStoSolver<T, K> {
  // Grants cereal access to default constructor/serialize functions
  friend class cereal::access;

 protected:
  using TStoSolver<T, K>::t;
  using TStoSolver<T, K>::model;
  using TStoSolver<T, K>::iterate;
  using TStoSolver<T, K>::prox;
  using TStoSolver<T, K>::epoch_size;
  using TStoSolver<T, K>::get_next_i;
  using TStoSolver<T, K>::rand_unif;

 public:
  using TStoSolver<T, K>::set_model;
  using TStoSolver<T, K>::get_minimizer;
  using TStoSolver<T, K>::set_starting_iterate;
  using TStoSolver<T, K>::get_class_name;

 protected:
  bool solver_ready = false;
  bool ready_step_corrections = false;
  T step = 0;
  // Probabilistic correction of the step-sizes of all model weights,
  // given by the inverse proportion of non-zero entries in each feature column
  Array<T> steps_correction;

  int record_every = 1;

  std::shared_ptr<TModelGeneralizedLinear<T, K>> casted_model;

  std::shared_ptr<TProxSeparable<T, K>> casted_prox;

  virtual void initialize_solver() = 0;

  void prepare_solve();

  void compute_step_corrections();

  virtual void solve_dense(bool use_intercept, ulong n_features) {
    TICK_CLASS_DOES_NOT_IMPLEMENT("BaseSAGA");
  }

  virtual void solve_sparse_proba_updates(bool use_intercept,
                                          ulong n_features) {
    TICK_CLASS_DOES_NOT_IMPLEMENT("BaseSAGA");
  }

 public:
  // This exists soley for cereal/swig
  TBaseSAGA() : TBaseSAGA<T, K>(0, 0, RandType::unif, 0, 0) {}

  TBaseSAGA(ulong epoch_size, T tol, RandType rand_type, T step, int seed);

  void solve() override;

  void set_model(std::shared_ptr<TModel<T, K>> model) override;

  T get_step() const { return step; }

  void set_step(T step) { this->step = step; }

  int get_record_every() const { return record_every; }
  void set_record_every(const int record_every) {
    this->record_every = record_every;
  }

 public:
  template <class Archive>
  void serialize(Archive &ar) {
    ar(cereal::make_nvp("StoSolver",
                        typename cereal::base_class<TStoSolver<T, K>>(this)));

    ar(CEREAL_NVP(step));
    ar(CEREAL_NVP(steps_correction));
    ar(CEREAL_NVP(solver_ready));
    ar(CEREAL_NVP(ready_step_corrections));
  }

  BoolStrReport compare(const TBaseSAGA<T, K> &that, std::stringstream &ss) {
    bool ret = TStoSolver<T, K>::compare(that, ss) &&
               TICK_CMP_REPORT(ss, step) &&
               TICK_CMP_REPORT(ss, steps_correction) &&
               TICK_CMP_REPORT(ss, solver_ready) &&
               TICK_CMP_REPORT(ss, ready_step_corrections);
    return BoolStrReport(ret, ss.str());
  }
};

using BaseSAGADouble = TBaseSAGA<double, double>;
CEREAL_SPECIALIZE_FOR_ALL_ARCHIVES(BaseSAGADouble,
                                   cereal::specialization::member_serialize)

using BaseSAGAFloat = TBaseSAGA<float, float>;
CEREAL_SPECIALIZE_FOR_ALL_ARCHIVES(BaseSAGAFloat,
                                   cereal::specialization::member_serialize)

using BaseSAGAAtomicDouble = TBaseSAGA<double, std::atomic<double>>;
CEREAL_SPECIALIZE_FOR_ALL_ARCHIVES(BaseSAGAAtomicDouble,
                                   cereal::specialization::member_serialize)

using BaseSAGAAtomicFloat = TBaseSAGA<float, std::atomic<float>>;
CEREAL_SPECIALIZE_FOR_ALL_ARCHIVES(BaseSAGAAtomicFloat,
                                   cereal::specialization::member_serialize)

template <class T>
class DLL_PUBLIC TSAGA : public TBaseSAGA<T, T> {
  // Grants cereal access to default constructor/serialize functions
  friend class cereal::access;

 protected:
  using TBaseSAGA<T, T>::get_next_i;
  using TBaseSAGA<T, T>::iterate;
  using TBaseSAGA<T, T>::steps_correction;
  using TBaseSAGA<T, T>::solve_dense;
  using TBaseSAGA<T, T>::solve_sparse_proba_updates;
  using TBaseSAGA<T, T>::model;
  using TBaseSAGA<T, T>::casted_model;
  using TBaseSAGA<T, T>::prox;
  using TBaseSAGA<T, T>::casted_prox;
  using TBaseSAGA<T, T>::epoch_size;
  using TBaseSAGA<T, T>::step;
  using TBaseSAGA<T, T>::t;
  using TBaseSAGA<T, T>::solver_ready;

 public:
  using TBaseSAGA<T, T>::set_starting_iterate;
  using TBaseSAGA<T, T>::get_minimizer;
  using TBaseSAGA<T, T>::set_model;
  using TBaseSAGA<T, T>::get_class_name;

 protected:
  Array<T> gradients_memory;
  Array<T> gradients_average;

  void initialize_solver() override;

  void solve_dense(bool use_intercept, ulong n_features) override;

  void solve_sparse_proba_updates(bool use_intercept,
                                  ulong n_features) override;

 public:
  // This exists soley for cereal/swig
  TSAGA() : TSAGA<T>(0, 0, RandType::unif, 0, 0) {}

  TSAGA(ulong epoch_size, T tol, RandType rand_type, T step, int seed = -1);

 public:
  template <class Archive>
  void serialize(Archive &ar) {
    ar(cereal::make_nvp("BaseSAGA", typename cereal::base_class<TBaseSAGA<T, T>>(this)));
    ar(CEREAL_NVP(gradients_memory));
    ar(CEREAL_NVP(gradients_average));
  }

  BoolStrReport compare(const TSAGA<T> &that) {
    std::stringstream ss;
    ss << get_class_name() << std::endl;
    auto is_equal = TBaseSAGA<T, T>::compare(that, ss) &&
                    TICK_CMP_REPORT(ss, gradients_memory) &&
                    TICK_CMP_REPORT(ss, gradients_average);
    return BoolStrReport(is_equal, ss.str());
  }

  BoolStrReport operator==(const TSAGA<T> &that) { return compare(that); }

  static std::shared_ptr<TSAGA<T>> AS_NULL() {
    return std::move(std::shared_ptr<TSAGA<T>>(new TSAGA<T>));
  }
};

using SAGA = TSAGA<double>;
using SAGADouble = TSAGA<double>;
CEREAL_SPECIALIZE_FOR_ALL_ARCHIVES(SAGADouble,
                                   cereal::specialization::member_serialize)
CEREAL_REGISTER_TYPE(SAGADouble)

using SAGAFloat = TSAGA<float>;
CEREAL_SPECIALIZE_FOR_ALL_ARCHIVES(SAGAFloat,
                                   cereal::specialization::member_serialize)
CEREAL_REGISTER_TYPE(SAGAFloat)

#endif  // LIB_INCLUDE_TICK_SOLVER_SAGA_H_
