#ifndef LIB_INCLUDE_TICK_SOLVER_SAGA_H_
#define LIB_INCLUDE_TICK_SOLVER_SAGA_H_

// License: BSD 3 clause

#include <atomic>
#include "sto_solver.h"
#include "tick/base_model/model_generalized_linear.h"

// T : float or double, type of feature arrays used
// K : atomic or not: type of iterate
// L : atomic or not: type of gradient memory and average
template <class T, class K, class L>
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

  using TStoSolver<T, K>::record_every;
  using TStoSolver<T, K>::save_history;
  using TStoSolver<T, K>::last_record_epoch;
  using TStoSolver<T, K>::last_record_time;
  using TStoSolver<T, K>::get_generator;

 public:
  using TStoSolver<T, K>::set_model;
  using TStoSolver<T, K>::get_minimizer;
  using TStoSolver<T, K>::set_starting_iterate;
  using TStoSolver<T, K>::get_class_name;

 protected:
  bool solver_ready = false;
  bool ready_step_corrections = false;
  T step = 0;
  uint n_threads = 1;
  // Probabilistic correction of the step-sizes of all model weights,
  // given by the inverse proportion of non-zero entries in each feature column
  Array<T> steps_correction;

  Array<L> gradients_memory;
  Array<L> gradients_average;

  std::shared_ptr<TModelGeneralizedLinear<T, K>> casted_model;

  std::shared_ptr<TProxSeparable<T, K>> casted_prox;

  virtual void initialize_solver();

  void prepare_solve();

  void compute_step_corrections();

 public:
  // This exists soley for cereal/swig
  TBaseSAGA() : TBaseSAGA<T, K, L>(0, 0, RandType::unif, 0, 0) {}

  TBaseSAGA(ulong epoch_size, T tol, RandType rand_type, T step, int record_every = 1,
      int seed = -1, int n_threads = 1);

  void set_model(std::shared_ptr<TModel<T, K>> model) override;
  void solve(int n_epochs = 1) override;

  T get_step() const { return step; }

  void set_step(T step) { this->step = step; }

 protected:
  virtual T update_gradient_memory(ulong i);
  virtual void update_iterate_and_gradient_average(ulong j, T x_ij, T grad_factor_diff,
                                                   T step_correction);

 public:
  template <class Archive>
  void serialize(Archive &ar) {
    ar(cereal::make_nvp("StoSolver",
                        typename cereal::base_class<TStoSolver<T, K>>(this)));

    ar(CEREAL_NVP(step));
    ar(CEREAL_NVP(steps_correction));
    ar(CEREAL_NVP(solver_ready));
    ar(CEREAL_NVP(ready_step_corrections));
    ar(CEREAL_NVP(gradients_memory));
    ar(CEREAL_NVP(gradients_average));
  }

  BoolStrReport compare(const TBaseSAGA<T, K, L> &that, std::stringstream &ss) {
    bool ret = TStoSolver<T, K>::compare(that, ss) &&
               TICK_CMP_REPORT(ss, step) &&
               TICK_CMP_REPORT(ss, steps_correction) &&
               TICK_CMP_REPORT(ss, solver_ready) &&
               TICK_CMP_REPORT(ss, ready_step_corrections) &&
               TICK_CMP_REPORT(ss, gradients_memory) &&
               TICK_CMP_REPORT(ss, gradients_average);
    return BoolStrReport(ret, ss.str());
  }

  BoolStrReport compare(const TBaseSAGA<T, K, L> &that) {
    std::stringstream ss;
    ss << get_class_name() << std::endl;

    return compare(that, ss);
  }

  BoolStrReport operator==(const TBaseSAGA<T, K, L> &that) { return compare(that); }
};

using BaseSAGADouble = TBaseSAGA<double, double, double>;
CEREAL_SPECIALIZE_FOR_ALL_ARCHIVES(BaseSAGADouble,
                                   cereal::specialization::member_serialize)

using BaseSAGADoubleAtomicIterate = TBaseSAGA<double, std::atomic<double>, double>;
CEREAL_SPECIALIZE_FOR_ALL_ARCHIVES(BaseSAGADoubleAtomicIterate,
                                   cereal::specialization::member_serialize)

using BaseSAGAFloat = TBaseSAGA<float, float, float>;
CEREAL_SPECIALIZE_FOR_ALL_ARCHIVES(BaseSAGAFloat,
                                   cereal::specialization::member_serialize)

using BaseSAGAAtomicDouble = TBaseSAGA<double, double, std::atomic<double>>;
CEREAL_SPECIALIZE_FOR_ALL_ARCHIVES(BaseSAGAAtomicDouble,
                                   cereal::specialization::member_serialize)

using BaseSAGAAtomicDoubleAtomicIterate  = TBaseSAGA<double, std::atomic<double>, std::atomic<double>>;
CEREAL_SPECIALIZE_FOR_ALL_ARCHIVES(BaseSAGAAtomicDoubleAtomicIterate ,
                                   cereal::specialization::member_serialize)

using BaseSAGAAtomicFloat = TBaseSAGA<float, float, std::atomic<float>>;
CEREAL_SPECIALIZE_FOR_ALL_ARCHIVES(BaseSAGAAtomicFloat,
                                   cereal::specialization::member_serialize)

template <class T, class K>
class DLL_PUBLIC TSAGA : public TBaseSAGA<T, K, T> {
  // Grants cereal access to default constructor/serialize functions
  friend class cereal::access;

 protected:
  using TBaseSAGA<T, K, T>::iterate;
  using TBaseSAGA<T, K, T>::model;
  using TBaseSAGA<T, K, T>::casted_prox;
  using TBaseSAGA<T, K, T>::step;
  using TBaseSAGA<T, K, T>::gradients_average;
  using TBaseSAGA<T, K, T>::gradients_memory;

  T update_gradient_memory(ulong i) override;
  void update_iterate_and_gradient_average(ulong j, T x_ij, T grad_factor_diff,
                                           T step_correction) override;

 public:
  // This exists soley for cereal/swig
  TSAGA() : TSAGA<T, K>(0, 0, RandType::unif, 0, 0) {}

  TSAGA(ulong epoch_size, T tol, RandType rand_type, T step, int record_every = 1, int seed = -1, int n_threads = 1);

  template <class Archive>
  void serialize(Archive &ar) {
    ar(cereal::make_nvp("BaseSAGA", typename cereal::base_class<TBaseSAGA<T, K, T>>(this)));
  }
};

using SAGA = TSAGA<double, double>;
using SAGADouble = TSAGA<double, double>;
CEREAL_SPECIALIZE_FOR_ALL_ARCHIVES(SAGADouble,
                                   cereal::specialization::member_serialize)
CEREAL_REGISTER_TYPE(SAGADouble)

using SAGADoubleAtomicIterate = TSAGA<double, std::atomic<double> >;
CEREAL_SPECIALIZE_FOR_ALL_ARCHIVES(SAGADoubleAtomicIterate,
                                   cereal::specialization::member_serialize)
CEREAL_REGISTER_TYPE(SAGADoubleAtomicIterate)

using SAGAFloat = TSAGA<float, float>;
CEREAL_SPECIALIZE_FOR_ALL_ARCHIVES(SAGAFloat,
                                   cereal::specialization::member_serialize)
CEREAL_REGISTER_TYPE(SAGAFloat)



template <class T, class K,
    std::memory_order M=std::memory_order_seq_cst,
    std::memory_order N=std::memory_order_seq_cst,
    bool O=true>
class DLL_PUBLIC AtomicSAGA : public TBaseSAGA<T, K, std::atomic<T>> {
  // Grants cereal access to default constructor/serialize functions
  friend class cereal::access;

 protected:
  using TBaseSAGA<T, K, std::atomic<T>>::iterate;
  using TBaseSAGA<T, K, std::atomic<T>>::model;
  using TBaseSAGA<T, K, std::atomic<T>>::casted_prox;
  using TBaseSAGA<T, K, std::atomic<T>>::step;

 public:
  using TBaseSAGA<T, K, std::atomic<T>>::gradients_average;
  using TBaseSAGA<T, K, std::atomic<T>>::gradients_memory;

 public:
  AtomicSAGA() : AtomicSAGA(0, 0, RandType::unif, 0) {}

  AtomicSAGA(ulong epoch_size, T tol, RandType rand_type, T step, int record_every = 1,
             int seed = -1, int n_threads = 2);

  ~AtomicSAGA() {}

 protected:
  T update_gradient_memory(ulong i) override;
  void update_iterate_and_gradient_average(ulong j, T x_ij, T grad_factor_diff,
                                           T step_correction) override;
};

using ASAGA = AtomicSAGA<double, double>;
using AtomicSAGADouble = AtomicSAGA<double, double>;
CEREAL_SPECIALIZE_FOR_ALL_ARCHIVES(AtomicSAGADouble,
                                   cereal::specialization::member_serialize)
CEREAL_REGISTER_TYPE(AtomicSAGADouble)

using AtomicSAGADoubleAtomicIterate = AtomicSAGA<double, std::atomic<double>>;
CEREAL_SPECIALIZE_FOR_ALL_ARCHIVES(AtomicSAGADoubleAtomicIterate,
                                   cereal::specialization::member_serialize)
CEREAL_REGISTER_TYPE(AtomicSAGADoubleAtomicIterate)

using AtomicSAGAFloat = AtomicSAGA<float, float>;
CEREAL_SPECIALIZE_FOR_ALL_ARCHIVES(AtomicSAGAFloat,
                                   cereal::specialization::member_serialize)
CEREAL_REGISTER_TYPE(AtomicSAGAFloat)



template <class T, class K,
    std::memory_order M,
    std::memory_order N>
class DLL_PUBLIC AtomicSAGA<T, K, M, N, false> : public TBaseSAGA<T, K, std::atomic<T>> {
  // Grants cereal access to default constructor/serialize functions
  friend class cereal::access;

 protected:
  using TBaseSAGA<T, K, std::atomic<T>>::iterate;
  using TBaseSAGA<T, K, std::atomic<T>>::model;
  using TBaseSAGA<T, K, std::atomic<T>>::casted_prox;
  using TBaseSAGA<T, K, std::atomic<T>>::step;

 public:
  using TBaseSAGA<T, K, std::atomic<T>>::gradients_average;
  using TBaseSAGA<T, K, std::atomic<T>>::gradients_memory;

 public:
  AtomicSAGA() : AtomicSAGA(0, 0, RandType::unif, 0) {}

  AtomicSAGA(ulong epoch_size, T tol, RandType rand_type, T step, int record_every = 1,
             int seed = -1, int n_threads = 2);

  ~AtomicSAGA() {}

 protected:
  T update_gradient_memory(ulong i) override;
  void update_iterate_and_gradient_average(ulong j, T x_ij, T grad_factor_diff,
                                           T step_correction) override;
};


using AtomicSAGARelax = AtomicSAGA<double, double, std::memory_order_relaxed, std::memory_order_relaxed>;
using AtomicSAGANoLoad = AtomicSAGA<double, double, std::memory_order_seq_cst, std::memory_order_seq_cst, false>;

#endif  // LIB_INCLUDE_TICK_SOLVER_SAGA_H_
