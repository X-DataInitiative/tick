//
// Created by Martin Bompaire on 12/09/2018.
//

#ifndef LIB_INCLUDE_TICK_SOLVER_ASAGA_H_
#define LIB_INCLUDE_TICK_SOLVER_ASAGA_H_

#include "tick/solver/saga.h"


template <class T>
class DLL_PUBLIC AtomicSAGA : public TBaseSAGA<T, T> {
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
  using TBaseSAGA<T, T>::record_every;

 public:
  using TBaseSAGA<T, T>::set_starting_iterate;
  using TBaseSAGA<T, T>::get_minimizer;
  using TBaseSAGA<T, T>::set_model;
  using TBaseSAGA<T, T>::get_class_name;

 private:
  int n_threads = 0;      // SWIG doesn't support uints
  size_t un_threads = 0;  //   uint == int = Werror
  ulong iterations;

  ArrayDouble objective, history;
  Array2d<T> iterates_history;

  Array<std::atomic<T>> gradients_memory;
  Array<std::atomic<T>> gradients_average;

  void initialize_solver() override;

 public:
  AtomicSAGA() : AtomicSAGA(0, 0, 0, RandType::unif, 0) {}

  AtomicSAGA(ulong epoch_size, ulong iterations, T tol, RandType rand_type,
             T step, int seed = -1, int n_threads = 2);

  ~AtomicSAGA() {}

  void solve_dense(bool use_intercept, ulong n_features) override;

  void solve_sparse_proba_updates(bool use_intercept,
                                  ulong n_features) override;

  void solve_sparse_thread(bool use_intercept, ulong n_features,
                           TProxSeparable<T, std::atomic<T>> *,
                           uint16_t thread_id);

  const ArrayDouble &get_objective() { return objective; }

  const ArrayDouble &get_computed_objective() {
    ulong n_records = std::ceil(static_cast<double>(iterations) / record_every);
    objective = ArrayDouble(n_records);
    for (ulong index = 0; index < n_records; ++index) {
      Array<T> iterate_index = view_row(iterates_history, index);
      objective[index] =
          model->loss(iterate_index) +
              prox->value(iterate_index, prox->get_start(), prox->get_end());
    }
    return objective;
  }

  const ArrayDouble &get_history() { return history; }

  template <class Archive>
  void load(Archive &ar) {
    ar(cereal::make_nvp("BaseSAGA", cereal::base_class<TBaseSAGA<T, T>>(this)));
    ar(CEREAL_NVP(n_threads));
    ar(CEREAL_NVP(un_threads));
    ar(CEREAL_NVP(iterations));
    ar(CEREAL_NVP(objective));
    ar(CEREAL_NVP(history));
    ar(CEREAL_NVP(iterates_history));

    Array<T> non_atomic_gradients_memory;
    ar(CEREAL_NVP(non_atomic_gradients_memory));
    gradients_memory = Array<std::atomic<T>>(non_atomic_gradients_memory.size());
    gradients_memory.init_to_zero();
    gradients_memory.mult_incr(non_atomic_gradients_memory, 1);

    Array<T> non_atomic_gradients_average;
    ar(CEREAL_NVP(non_atomic_gradients_average));
    gradients_average = Array<std::atomic<T>>(non_atomic_gradients_average.size());
    gradients_average.init_to_zero();
    gradients_average.mult_incr(non_atomic_gradients_average, 1);
  }

  template <class Archive>
  void save(Archive &ar) const {
    ar(cereal::make_nvp("BaseSAGA", cereal::base_class<TBaseSAGA<T, T>>(this)));
    ar(CEREAL_NVP(n_threads));
    ar(CEREAL_NVP(un_threads));
    ar(CEREAL_NVP(iterations));
    ar(CEREAL_NVP(objective));
    ar(CEREAL_NVP(history));
    ar(CEREAL_NVP(iterates_history));

    Array<T> non_atomic_gradients_memory(gradients_memory.size());
    non_atomic_gradients_memory.init_to_zero();
    non_atomic_gradients_memory.mult_incr(non_atomic_gradients_memory, 1);
    ar(CEREAL_NVP(non_atomic_gradients_memory));

    Array<T> non_atomic_gradients_average(gradients_average.size());
    non_atomic_gradients_average.init_to_zero();
    non_atomic_gradients_average.mult_incr(non_atomic_gradients_average, 1);
    ar(CEREAL_NVP(non_atomic_gradients_average));
  }

  BoolStrReport compare(const AtomicSAGA<T> &that) {
    std::stringstream ss;
    ss << get_class_name() << std::endl;
    auto is_equal = TBaseSAGA<T, T>::compare(that, ss) &&
        TICK_CMP_REPORT(ss, n_threads) &&
        TICK_CMP_REPORT(ss, un_threads) &&
        TICK_CMP_REPORT(ss, iterations) &&
        TICK_CMP_REPORT(ss, objective) &&
        TICK_CMP_REPORT(ss, history) &&
        TICK_CMP_REPORT(ss, iterates_history) &&
        TICK_CMP_REPORT(ss, gradients_memory) &&
        TICK_CMP_REPORT(ss, gradients_average);
    return BoolStrReport(is_equal, ss.str());
  }

  BoolStrReport operator==(const AtomicSAGA<T> &that) { return compare(that); }

  static std::shared_ptr<AtomicSAGA<T>> AS_NULL() {
    return std::move(std::shared_ptr<AtomicSAGA<T>>(new AtomicSAGA<T>));
  }
};

using ASAGA = AtomicSAGA<double>;
using AtomicSAGADouble = AtomicSAGA<double>;
CEREAL_SPECIALIZE_FOR_ALL_ARCHIVES(AtomicSAGADouble,
                                   cereal::specialization::member_load_save)
CEREAL_REGISTER_TYPE(AtomicSAGADouble)
using AtomicSAGAFloat = AtomicSAGA<float>;
CEREAL_SPECIALIZE_FOR_ALL_ARCHIVES(AtomicSAGAFloat,
                                   cereal::specialization::member_load_save)
CEREAL_REGISTER_TYPE(AtomicSAGAFloat)

#endif  // LIB_INCLUDE_TICK_SOLVER_ASAGA_H_
