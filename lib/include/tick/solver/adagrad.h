#ifndef LIB_INCLUDE_TICK_SOLVER_ADAGRAD_H_
#define LIB_INCLUDE_TICK_SOLVER_ADAGRAD_H_

// License: BSD 3 clause

#include "sto_solver.h"

template <class T>
class DLL_PUBLIC TAdaGrad : public TStoSolver<T> {
  // Grants cereal access to default constructor
  friend class cereal::access;

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
  Array<T> hist_grad;
  T step;

 public:
  // This exists soley for cereal/swig
  TAdaGrad() : TAdaGrad<T>(0, 0, RandType::unif, 0, 0) {}

  TAdaGrad(ulong epoch_size, T tol, RandType rand_type, T step, int record_every = 1,
      int seed = -1);

  void solve_one_epoch() override;

  void set_starting_iterate(Array<T> &new_iterate) override;

  template <class Archive>
  void serialize(Archive &ar) {
    ar(cereal::make_nvp("StoSolver", cereal::base_class<TStoSolver<T> >(this)));

    ar(CEREAL_NVP(hist_grad));
    ar(CEREAL_NVP(step));
  }

  BoolStrReport compare(const TAdaGrad<T> &that) {
    std::stringstream ss;
    ss << get_class_name() << std::endl;
    bool are_equal = TStoSolver<T>::compare(that, ss) &&
                     TICK_CMP_REPORT(ss, hist_grad) &&
                     TICK_CMP_REPORT(ss, step);
    return BoolStrReport(are_equal, ss.str());
  }

  BoolStrReport operator==(const TAdaGrad<T> &that) { return compare(that); }

  static std::shared_ptr<TAdaGrad<T> > AS_NULL() {
    return std::move(std::shared_ptr<TAdaGrad<T> >(new TAdaGrad<T>));
  }
};

using AdaGrad = TAdaGrad<double>;

using AdaGradDouble = TAdaGrad<double>;
CEREAL_SPECIALIZE_FOR_ALL_ARCHIVES(AdaGradDouble,
                                   cereal::specialization::member_serialize)
CEREAL_REGISTER_TYPE(AdaGradDouble)

using AdaGradFloat = TAdaGrad<float>;
CEREAL_SPECIALIZE_FOR_ALL_ARCHIVES(AdaGradFloat,
                                   cereal::specialization::member_serialize)
CEREAL_REGISTER_TYPE(AdaGradFloat)

#endif  // LIB_INCLUDE_TICK_SOLVER_ADAGRAD_H_
