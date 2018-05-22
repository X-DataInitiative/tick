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

 public:
  enum class VarianceReductionMethod {
    Last = 1,
    Average = 2,
    Random = 3,
  };

 protected:
  bool solver_ready = false;
  bool ready_step_corrections = false;
  uint64_t rand_index = 0;
  T step = 0;
  // Probabilistic correction of the step-sizes of all model weights,
  // given by the inverse proportion of non-zero entries in each feature column
  Array<T> steps_correction;

  SAGA_VarianceReductionMethod variance_reduction;

  Array<K> next_iterate;

  Array<K> gradients_memory;
  Array<K> gradients_average;

  std::shared_ptr<TModelGeneralizedLinear<T, K>> casted_model;

  std::shared_ptr<TProxSeparable<T, K>> casted_prox;

  void initialize_solver();

  void prepare_solve();

  void compute_step_corrections();

  void set_starting_iterate(Array<K> &new_iterate) override;

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

  TBaseSAGA(ulong epoch_size, T tol, RandType rand_type, T step, int seed,
            SAGA_VarianceReductionMethod variance_reduction =
                SAGA_VarianceReductionMethod::Last);

  void solve() override;

  void set_model(std::shared_ptr<TModel<T, K>> model) override;

  T get_step() const { return step; }

  void set_step(T step) { this->step = step; }

  SAGA_VarianceReductionMethod get_variance_reduction() const {
    return variance_reduction;
  }

  void set_variance_reduction(
      SAGA_VarianceReductionMethod _variance_reduction) {
    variance_reduction = _variance_reduction;
  }

 public:
  template <class Archive>
  void serialize(Archive &ar) {
    ar(cereal::make_nvp("StoSolver",
                        typename cereal::base_class<TStoSolver<T, K>>(this)));

    ar(CEREAL_NVP(step));
    ar(CEREAL_NVP(steps_correction));
    ar(CEREAL_NVP(variance_reduction));
    ar(CEREAL_NVP(next_iterate));
    ar(CEREAL_NVP(solver_ready));
    ar(CEREAL_NVP(gradients_memory));
    ar(CEREAL_NVP(gradients_average));
    ar(CEREAL_NVP(rand_index));
    ar(CEREAL_NVP(ready_step_corrections));
  }

  BoolStrReport compare(const TBaseSAGA<T, K> &that, std::stringstream &ss) {
    bool ret = TStoSolver<T, K>::compare(that, ss) &&
               TICK_CMP_REPORT(ss, step) &&
               TICK_CMP_REPORT(ss, steps_correction) &&
               TICK_CMP_REPORT(ss, variance_reduction) &&
               TICK_CMP_REPORT(ss, next_iterate) &&
               TICK_CMP_REPORT(ss, solver_ready) &&
               TICK_CMP_REPORT(ss, gradients_memory) &&
               TICK_CMP_REPORT(ss, gradients_average) &&
               TICK_CMP_REPORT(ss, rand_index) &&
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
  using TBaseSAGA<T, T>::next_iterate;
  using TBaseSAGA<T, T>::rand_index;
  using TBaseSAGA<T, T>::steps_correction;
  using TBaseSAGA<T, T>::gradients_average;
  using TBaseSAGA<T, T>::gradients_memory;
  using TBaseSAGA<T, T>::variance_reduction;
  using TBaseSAGA<T, T>::solve_dense;
  using TBaseSAGA<T, T>::solve_sparse_proba_updates;
  using TBaseSAGA<T, T>::model;
  using TBaseSAGA<T, T>::casted_model;
  using TBaseSAGA<T, T>::prox;
  using TBaseSAGA<T, T>::casted_prox;
  using TBaseSAGA<T, T>::epoch_size;
  using TBaseSAGA<T, T>::step;
  using TBaseSAGA<T, T>::t;

 public:
  using TBaseSAGA<T, T>::set_variance_reduction;
  using TBaseSAGA<T, T>::set_starting_iterate;
  using TBaseSAGA<T, T>::get_minimizer;
  using TBaseSAGA<T, T>::set_model;
  using TBaseSAGA<T, T>::get_class_name;

 protected:
  virtual void solve_dense(bool use_intercept, ulong n_features);

  virtual void solve_sparse_proba_updates(bool use_intercept, ulong n_features);

 public:
  // This exists soley for cereal/swig
  TSAGA() : TSAGA<T>(0, 0, RandType::unif, 0, 0) {}

  TSAGA(ulong epoch_size, T tol, RandType rand_type, T step, int seed = -1,
        SAGA_VarianceReductionMethod variance_reduction =
            SAGA_VarianceReductionMethod::Last);

 public:
  template <class Archive>
  void serialize(Archive &ar) {
    ar(cereal::make_nvp("BaseSAGA",
                        typename cereal::base_class<TBaseSAGA<T, T>>(this)));
  }

  BoolStrReport compare(const TSAGA<T> &that) {
    std::stringstream ss;
    ss << get_class_name() << std::endl;
    auto is_equal = TBaseSAGA<T, T>::compare(that, ss);
    return BoolStrReport(is_equal, ss.str());
  }

  BoolStrReport operator==(const TSAGA<T> &that) { return compare(that); }

  static std::shared_ptr<TSAGA<T>> AS_NULL() {
    return std::move(std::shared_ptr<TSAGA<T>>(new TSAGA<T>));
  }
};

using SAGADouble = TSAGA<double>;
CEREAL_SPECIALIZE_FOR_ALL_ARCHIVES(SAGADouble,
                                   cereal::specialization::member_serialize)
CEREAL_REGISTER_TYPE(SAGADouble)

using SAGAFloat = TSAGA<float>;
CEREAL_SPECIALIZE_FOR_ALL_ARCHIVES(SAGAFloat,
                                   cereal::specialization::member_serialize)
CEREAL_REGISTER_TYPE(SAGAFloat)

template <class T>
class AtomicSAGA : public TBaseSAGA<T, std::atomic<T>> {
  // Grants cereal access to default constructor/serialize functions
  friend class cereal::access;

 protected:
  using TBaseSAGA<T, std::atomic<T>>::get_next_i;
  using TBaseSAGA<T, std::atomic<T>>::iterate;
  using TBaseSAGA<T, std::atomic<T>>::next_iterate;
  using TBaseSAGA<T, std::atomic<T>>::rand_index;
  using TBaseSAGA<T, std::atomic<T>>::steps_correction;
  using TBaseSAGA<T, std::atomic<T>>::gradients_average;
  using TBaseSAGA<T, std::atomic<T>>::gradients_memory;
  using TBaseSAGA<T, std::atomic<T>>::variance_reduction;
  using TBaseSAGA<T, std::atomic<T>>::solve_dense;
  using TBaseSAGA<T, std::atomic<T>>::solve_sparse_proba_updates;
  using TBaseSAGA<T, std::atomic<T>>::model;
  using TBaseSAGA<T, std::atomic<T>>::casted_model;
  using TBaseSAGA<T, std::atomic<T>>::prox;
  using TBaseSAGA<T, std::atomic<T>>::casted_prox;
  using TBaseSAGA<T, std::atomic<T>>::epoch_size;
  using TBaseSAGA<T, std::atomic<T>>::step;
  using TBaseSAGA<T, std::atomic<T>>::t;

 public:
  using TBaseSAGA<T, std::atomic<T>>::set_variance_reduction;
  using TBaseSAGA<T, std::atomic<T>>::set_starting_iterate;
  using TBaseSAGA<T, std::atomic<T>>::get_minimizer;
  using TBaseSAGA<T, std::atomic<T>>::set_model;
  using TBaseSAGA<T, std::atomic<T>>::get_class_name;

 private:
  int n_threads = 0;      // SWIG doesn't support uints
  size_t un_threads = 0;  //   uint == int = Werror
  ulong iterations;

  ArrayDouble objective, history;

 public:
  AtomicSAGA() : AtomicSAGA(0, 0, 0, RandType::unif, 0) {}

  AtomicSAGA(ulong epoch_size, ulong iterations, T tol, RandType rand_type,
             T step, int seed = -1,
             SAGA_VarianceReductionMethod variance_reduction =
                 SAGA_VarianceReductionMethod::Last,
             int n_threads = 2);

  void solve_dense(bool use_intercept, ulong n_features) override;

  void solve_sparse_proba_updates(bool use_intercept,
                                  ulong n_features) override;
  void solve_sparse_thread(bool use_intercept, ulong n_features,
                           TProxSeparable<T, std::atomic<T>> *,
                           uint16_t thread_id);

  void get_atomic_minimizer(Array<std::atomic<T>> &out);

  const ArrayDouble &get_objective() { return objective; }
  const ArrayDouble &get_history() { return history; }

  // disabled for the moment
  template <class Archive>
  void serialize(Archive &ar) {
    ar(cereal::make_nvp(
        "BaseSAGA", cereal::base_class<TBaseSAGA<T, std::atomic<T>>>(this)));
    ar(n_threads);
    ar(un_threads);
    ar(iterations);
    ar(objective);
    ar(history);
  }

  BoolStrReport compare(const AtomicSAGA<T> &that) {
    std::stringstream ss;
    ss << get_class_name() << std::endl;
    auto is_equal = TBaseSAGA<T, std::atomic<T>>::compare(that, ss);
    return BoolStrReport(is_equal, ss.str());
  }

  BoolStrReport operator==(const AtomicSAGA<T> &that) { return compare(that); }

  static std::shared_ptr<AtomicSAGA<T>> AS_NULL() {
    return std::move(std::shared_ptr<AtomicSAGA<T>>(new AtomicSAGA<T>));
  }
};

// using AtomicSAGADouble = AtomicSAGA<double>;
// CEREAL_SPECIALIZE_FOR_ALL_ARCHIVES(AtomicSAGADouble,
//                                    cereal::specialization::member_serialize)
// CEREAL_REGISTER_TYPE(AtomicSAGADouble)

// using AtomicSAGAFloat = AtomicSAGA<float>;
// CEREAL_SPECIALIZE_FOR_ALL_ARCHIVES(AtomicSAGAFloat,
//                                    cereal::specialization::member_serialize)
// CEREAL_REGISTER_TYPE(AtomicSAGAFloat)

#endif  // LIB_INCLUDE_TICK_SOLVER_SAGA_H_
