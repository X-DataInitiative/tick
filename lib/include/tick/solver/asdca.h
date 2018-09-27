//
// Created by Martin Bompaire on 22/10/15.
//

#ifndef LIB_INCLUDE_TICK_SOLVER_ASDCA_H_
#define LIB_INCLUDE_TICK_SOLVER_ASDCA_H_

// License: BSD 3 clause

#include "tick/solver/sdca.h"
#include "tick/base_model/model.h"

// TODO: profile the code of SDCA to check if it's faster
// TODO: code accelerated SDCA

template <class T>
 class DLL_PUBLIC AtomicSDCA : public TBaseSDCA<T, std::atomic<T> > {
 protected:
  using TBaseSDCA<T, std::atomic<T>>::t;
  using TBaseSDCA<T, std::atomic<T>>::model;
  using TBaseSDCA<T, std::atomic<T>>::iterate;
  using TBaseSDCA<T, std::atomic<T>>::prox;
  using TBaseSDCA<T, std::atomic<T>>::get_next_i;
  using TBaseSDCA<T, std::atomic<T>>::epoch_size;
  using TBaseSDCA<T, std::atomic<T>>::rand_max;
  using TBaseSDCA<T, std::atomic<T>>::stored_variables_ready;
  using TBaseSDCA<T, std::atomic<T>>::get_scaled_l_l2sq;
  using TBaseSDCA<T, std::atomic<T>>::delta;
  using TBaseSDCA<T, std::atomic<T>>::dual_vector;
  using TBaseSDCA<T, std::atomic<T>>::tmp_primal_vector;
  using TBaseSDCA<T, std::atomic<T>>::set_starting_iterate;
  using TBaseSDCA<T, std::atomic<T>>::casted_prox;
   using TBaseSDCA<T, std::atomic<T>>::n_threads;

  using TStoSolver<T, std::atomic<T>>::save_history;
  using TStoSolver<T, std::atomic<T>>::last_record_epoch;
  using TStoSolver<T, std::atomic<T>>::last_record_time;
  using TStoSolver<T, std::atomic<T>>::record_every;



 public:
  using TBaseSDCA<T, std::atomic<T>>::get_class_name;
  using SArrayTPtr = std::shared_ptr<SArray<T>>;

 public:
  // This exists solely for cereal/swig
  AtomicSDCA() : AtomicSDCA<T>(0, 0, 0) {}

  explicit AtomicSDCA(T l_l2sq, ulong epoch_size = 0, T tol = 0.,
                      RandType rand_type = RandType::unif,  int record_every = 1, int seed = -1,
                      int n_threads=2);

//  void solve(int n_epochs = 1) override;
   void update_delta_dual_i(ulong i, double delta_dual_i,
                            const BaseArray<T> &feature_i, double _1_over_lbda_n) override ;
  void solve_batch(int n_epochs = 1, ulong bach_size = 2);

//  void set_starting_iterate();
//  void set_starting_iterate(Array<std::atomic<T>> &dual_vector) override;

// public:
//  template <class Archive>
//  void serialize(Archive &ar) {
//    ar(cereal::make_nvp("StoSolver",
//                        cereal::base_class<TBaseSDCA<T, std::atomic<T>>>(this)));
//
//    ar(CEREAL_NVP(n_coeffs));
//    ar(CEREAL_NVP(stored_variables_ready));
//    ar(CEREAL_NVP(l_l2sq));
//    ar(CEREAL_NVP(delta));
//    ar(CEREAL_NVP(dual_vector));
//  }
//
//   BoolStrReport compare(const AtomicSDCA<T> &that) {
//     std::stringstream ss;
//     ss << get_class_name() << std::endl;
//     bool are_equal =
//         TBaseSDCA<T, std::atomic<T>>::compare(that, ss) && TICK_CMP_REPORT(ss, n_coeffs) &&
//         TICK_CMP_REPORT(ss, stored_variables_ready) &&
//         TICK_CMP_REPORT(ss, l_l2sq) && TICK_CMP_REPORT(ss, delta) &&
//         TICK_CMP_REPORT(ss, dual_vector);
//     return BoolStrReport(are_equal, ss.str());
//   }
//
//   BoolStrReport operator==(const AtomicSDCA<T> &that) { return compare(that); }
//
//   static std::shared_ptr<AtomicSDCA<T>> AS_NULL() {
//     return std::move(std::shared_ptr<AtomicSDCA<T>>(new AtomicSDCA<T>));
//   }
};

using ASDCA = AtomicSDCA<double>;

using ASDCADouble = AtomicSDCA<double>;
//CEREAL_SPECIALIZE_FOR_ALL_ARCHIVES(SDCADouble,
//                                   cereal::specialization::member_serialize)
//CEREAL_REGISTER_TYPE(SDCADouble)

using ASDCAFloat = AtomicSDCA<float>;
//CEREAL_SPECIALIZE_FOR_ALL_ARCHIVES(SDCAFloat,
//                                   cereal::specialization::member_serialize)
//CEREAL_REGISTER_TYPE(SDCAFloat)

#endif  // LIB_INCLUDE_TICK_SOLVER_ASDCA_H_
