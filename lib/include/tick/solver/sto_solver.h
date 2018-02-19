//
// Created by Martin Bompaire on 22/10/15.
//

#ifndef LIB_INCLUDE_TICK_SOLVER_STO_SOLVER_H_
#define LIB_INCLUDE_TICK_SOLVER_STO_SOLVER_H_

// License: BSD 3 clause

#include "tick/base/base.h"
#include "tick/base_model/model.h"
#include "tick/prox/prox.h"
#include "tick/prox/prox_zero.h"
#include "tick/random/rand.h"

// TODO: code an abstract class and use it for StoSolvers
// TODO: StoSolver and LabelsFeaturesSolver

// Type of randomness used when sampling at random data points
enum class RandType { unif = 0, perm };

template <class T>
class DLL_PUBLIC TStoSolver {
 protected:
  // A flag that specify if random permutation is ready to be used or not
  bool permutation_ready;

  // Seed of the random sampling
  int seed = -1;

  // Iteration counter
  ulong t = 1;

  // sampling is done in {0, ..., rand_max-1}
  // This is useful to know in what range random sampling must be done
  // (n_samples for generalized linear models, n_nodes for Hawkes processes,
  // etc.)
  ulong rand_max;

  // Number of steps within an epoch
  ulong epoch_size;

  // Current index in the permutation (useful when using random permutation
  // sampling)
  ulong i_perm;

  // Tolerance for convergence. Not used yet.
  T tol;

  // Model object
  std::shared_ptr<TModel<T> > model;

  std::shared_ptr<TProx<T> > prox;

  // Type of random sampling
  RandType rand_type;

  Rand rand;

  // Iterate
  Array<T> iterate;

  // An array that allows to store the sampled random permutation
  ArrayULong permutation;

 protected:
  // Init permutation array in case of Random is srt to permutation
  void init_permutation();

 public:
  inline TStoSolver(ulong epoch_size = 0, T tol = 0.,
                    RandType rand_type = RandType::unif, int seed = -1)
      : epoch_size(epoch_size),
        tol(tol),
        prox(std::make_shared<TProxZero<T> >(0.0)),
        rand_type(rand_type) {
    set_seed(seed);
    permutation_ready = false;
  }

  virtual ~TStoSolver() {}

  virtual void set_model(std::shared_ptr<TModel<T> > _model) {
    this->model = _model;
    permutation_ready = false;
    iterate = Array<T>(_model->get_n_coeffs());
    iterate.init_to_zero();
  }

  virtual void set_prox(std::shared_ptr<TProx<T> > prox) { this->prox = prox; }

  void set_seed(int seed) {
    this->seed = seed;
    rand = Rand(seed);
  }

  virtual void reset();

  ulong get_next_i();

  void shuffle();

  virtual void solve() { TICK_CLASS_DOES_NOT_IMPLEMENT("TStoSolver<T>"); }

  virtual void get_minimizer(Array<T> &out);

  virtual void get_iterate(Array<T> &out);

  virtual void set_starting_iterate(Array<T> &new_iterate);

  // Returns a uniform integer in the set {0, ..., m - 1}
  inline ulong rand_unif(ulong m) { return rand.uniform_int(ulong{0}, m); }

  inline T get_tol() const { return tol; }

  inline void set_tol(T tol) { this->tol = tol; }

  inline ulong get_epoch_size() const { return epoch_size; }

  inline void set_epoch_size(ulong epoch_size) {
    this->epoch_size = epoch_size;
  }

  inline ulong get_t() const { return t; }

  inline RandType get_rand_type() const { return rand_type; }

  inline void set_rand_type(RandType rand_type) { this->rand_type = rand_type; }

  inline ulong get_rand_max() const { return rand_max; }

  inline void set_rand_max(ulong rand_max) {
    this->rand_max = rand_max;
    permutation_ready = false;
  }
};

using StoSolver = TStoSolver<double>;
using TStoSolverDouble = TStoSolver<double>;
using TStoSolverFloat = TStoSolver<float>;

#endif  // LIB_INCLUDE_TICK_SOLVER_STO_SOLVER_H_
