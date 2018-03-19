//
// Created by Martin Bompaire on 22/10/15.
//

#ifndef LIB_INCLUDE_TICK_SOLVER_SGD_H_
#define LIB_INCLUDE_TICK_SOLVER_SGD_H_

// License: BSD 3 clause

#include "sto_solver.h"
#include "tick/base_model/model.h"
#include "tick/prox/prox.h"

template <class T>
class DLL_PUBLIC TSGD : public TStoSolver<T> {
 protected:
  using TStoSolver<T>::t;
  using TStoSolver<T>::model;
  using TStoSolver<T>::iterate;
  using TStoSolver<T>::prox;
  using TStoSolver<T>::epoch_size;
  using TStoSolver<T>::get_next_i;

 public:
  using TStoSolver<T>::get_class_name;

 private:
  T step_t;
  T step;

 public:
  TSGD(ulong epoch_size = 0, T tol = 0., RandType rand_type = RandType::unif,
       T step = 0., int seed = -1);

  inline T get_step_t() const { return step_t; }

  inline T get_step() const { return step; }

  inline void set_step(T step) { this->step = step; }

  void solve();

  void solve_sparse();

  inline T get_step_t();

  template <class Archive>
  void serialize(Archive &ar) {
    ar(cereal::make_nvp("StoSolver", cereal::base_class<TStoSolver<T>>(this)));

    ar(CEREAL_NVP(step_t));
    ar(CEREAL_NVP(step));
  }

  BoolStrReport compare(const TSGD<T> &that) {
    std::stringstream ss;
    ss << get_class_name() << std::endl;
    bool are_equal = TStoSolver<T>::compare(that, ss) &&
                     TICK_CMP_REPORT(ss, step_t) && TICK_CMP_REPORT(ss, step);
    return BoolStrReport(are_equal, ss.str());
  }

  BoolStrReport operator==(const TSGD<T> &that) { return compare(that); }

  static std::shared_ptr<TSGD<T>> AS_NULL() {
    return std::move(std::shared_ptr<TSGD<T>>(new TSGD<T>));
  }
};

using SGD = TSGD<double>;

using SGDDouble = TSGD<double>;
CEREAL_SPECIALIZE_FOR_ALL_ARCHIVES(SGDDouble,
                                   cereal::specialization::member_serialize)
CEREAL_REGISTER_TYPE(SGDDouble)

using SGDFloat = TSGD<float>;
CEREAL_SPECIALIZE_FOR_ALL_ARCHIVES(SGDFloat,
                                   cereal::specialization::member_serialize)
CEREAL_REGISTER_TYPE(SGDFloat)

#endif  // LIB_INCLUDE_TICK_SOLVER_SGD_H_
