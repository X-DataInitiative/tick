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
 class DLL_PUBLIC TAtomicSDCA : public TBaseSDCA<T, std::atomic<T> > {
 protected:
  using TBaseSDCA<T, std::atomic<T>>::model;
  using TBaseSDCA<T, std::atomic<T>>::delta;
  using TBaseSDCA<T, std::atomic<T>>::dual_vector;
  using TBaseSDCA<T, std::atomic<T>>::tmp_primal_vector;

 public:
  // This exists solely for cereal/swig
  TAtomicSDCA() : TAtomicSDCA<T>(0, 0, 0) {}

  explicit TAtomicSDCA(T l_l2sq, ulong epoch_size = 0, T tol = 0.,
                      RandType rand_type = RandType::unif,  int record_every = 1, int seed = -1,
                      int n_threads=2);

//  void solve(int n_epochs = 1) override;
   void update_delta_dual_i(ulong i, double delta_dual_i,
                            const BaseArray<T> &feature_i, double _1_over_lbda_n) override ;

};

using AtomicSDCA = TAtomicSDCA<double>;

using AtomicSDCADouble = TAtomicSDCA<double>;
//CEREAL_SPECIALIZE_FOR_ALL_ARCHIVES(SDCADouble,
//                                   cereal::specialization::member_serialize)
//CEREAL_REGISTER_TYPE(SDCADouble)

using AtomicSDCAFloat = TAtomicSDCA<float>;
//CEREAL_SPECIALIZE_FOR_ALL_ARCHIVES(SDCAFloat,
//                                   cereal::specialization::member_serialize)
//CEREAL_REGISTER_TYPE(SDCAFloat)

#endif  // LIB_INCLUDE_TICK_SOLVER_ASDCA_H_
