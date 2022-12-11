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

#include <iostream>
#include <sstream>

#include "tick/solver/enums.h"

// TODO: code an abstract class and use it for StoSolvers
// TODO: StoSolver and LabelsFeaturesSolver

// Type of randomness used when sampling at random data points
enum class RandType { unif = 0, perm };
inline std::ostream &operator<<(std::ostream &s, const RandType &r) {
  typedef std::underlying_type<RandType>::type utype;
  return s << static_cast<utype>(r);
}

template <class T, class K = T>
class DLL_PUBLIC TStoSolver {
  // Grants cereal access to default constructor/serialize functions
  friend class cereal::access;

  template <class T1, class K1>
  friend std::ostream &operator<<(std::ostream &, const TStoSolver<T1, K1> &);

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
  std::shared_ptr<TModel<T, K> > model;

  std::shared_ptr<TProx<T, K> > prox;

  // Type of random sampling
  RandType rand_type;

  Rand rand;

  // Iterate
  Array<K> iterate;

  // An array that allows to store the sampled random permutation
  ArrayULong permutation;

  int record_every = 1;
  size_t last_record_epoch = 0;
  double last_record_time = 0;
  // used for tolerance/history
  double first_obj = 0, prev_obj = 0;

  // A vector storing all timings at which history has been stored
  std::vector<double> time_history;
  std::vector<double> objectives;

  // A vector storing all epoch at which history has been stored
  std::vector<int> epoch_history;

  // A vector storing all timings at which history has been stored
  std::vector<Array<T> > iterate_history;

 protected:
  // Init permutation array in case of Random is srt to permutation
  void init_permutation();

  virtual void save_history(double time, int epoch);

 public:
  inline TStoSolver(ulong epoch_size = 0, T tol = 0., RandType rand_type = RandType::unif,
                    int record_every = 1, int seed = -1)
      : epoch_size(epoch_size),
        tol(tol),
        prox(std::make_shared<TProxZero<T, K> >(0.0)),
        rand_type(rand_type),
        record_every(record_every) {
    set_seed(seed);
    permutation_ready = false;
  }

  virtual ~TStoSolver() {}

  virtual void set_model(std::shared_ptr<TModel<T, K> > _model) {
    this->model = _model;
    permutation_ready = false;
    iterate = Array<K>(_model->get_n_coeffs());
    iterate.init_to_zero();
  }

  virtual void set_prox(std::shared_ptr<TProx<T, K> > prox) { this->prox = prox; }

  void set_seed(int seed) {
    this->seed = seed;
    rand = Rand(seed);
  }

  virtual void reset();

  ulong get_next_i();

  void shuffle();

  virtual void solve_one_epoch() { TICK_CLASS_DOES_NOT_IMPLEMENT("TStoSolver<T, K>"); }

  virtual void solve(size_t n_epochs = 1);

  virtual void get_minimizer(Array<T> &out);

  virtual void get_iterate(Array<T> &out);

  virtual void set_starting_iterate(Array<K> &new_iterate);

  // Returns a uniform integer in the set {0, ..., m - 1}
  inline ulong rand_unif(ulong m) {
    /*
     * Notice that
     * ulong Rand::uniform_ulong(ulong a, ulong b)
     * generates random unsigned long integers
     * uniformly distributed in the set
     * {a, ..., b}
     * with a and b included
     */
    return rand.uniform_ulong(ulong{0}, m - 1);
  }

  inline T get_tol() const { return tol; }

  inline void set_tol(T tol) { this->tol = tol; }

  inline ulong get_epoch_size() const { return epoch_size; }

  inline void set_epoch_size(ulong epoch_size) { this->epoch_size = epoch_size; }

  inline ulong get_t() const { return t; }

  inline RandType get_rand_type() const { return rand_type; }

  inline void set_rand_type(RandType rand_type) { this->rand_type = rand_type; }

  inline ulong get_rand_max() const { return rand_max; }

  inline void set_rand_max(ulong rand_max) {
    this->rand_max = rand_max;
    permutation_ready = false;
  }

  inline int get_record_every() const { return record_every; }

  inline void set_record_every(int record_every) { this->record_every = record_every; }

  std::vector<double> get_time_history() const { return time_history; }
  std::vector<double> get_objectives() const { return objectives; }
  double get_objective() const { return model->loss(iterate) + prox->value(iterate); }
  TStoSolver<T, K> &set_prev_obj(const double obj) {
    prev_obj = obj;
    return *this;
  }
  TStoSolver<T, K> &set_first_obj(const double obj) {
    first_obj = obj;
    return *this;
  }
  double get_first_obj() const { return first_obj; }

  std::vector<int> get_epoch_history() const { return epoch_history; }

  std::vector<std::shared_ptr<SArray<T> > > get_iterate_history() const;

  const std::shared_ptr<TModel<T, K> > get_model() { return model; }
  const std::shared_ptr<TProx<T, K> > get_prox() { return prox; }

  virtual std::string get_class_name() const {
    std::stringstream ss;
    ss << typeid(*this).name() << "<" << typeid(T).name() << ">";
    return ss.str();
  }

  template <class Archive>
  void load(Archive &ar) {
    ar(CEREAL_NVP(model));
    ar(CEREAL_NVP(prox));
    ar(CEREAL_NVP(t));
    ar(CEREAL_NVP(iterate));
    ar(CEREAL_NVP(rand_max));
    ar(CEREAL_NVP(epoch_size));
    ar(CEREAL_NVP(tol));
    ar(CEREAL_NVP(rand_type));
    ar(CEREAL_NVP(permutation));
    ar(CEREAL_NVP(i_perm));
    ar(CEREAL_NVP(permutation_ready));
    ar(CEREAL_NVP(record_every));
    int rand_seed;
    ar(CEREAL_NVP(rand_seed));
    rand = Rand(rand_seed);
  }

  template <class Archive>
  void save(Archive &ar) const {
    ar(CEREAL_NVP(model));
    ar(CEREAL_NVP(prox));
    ar(CEREAL_NVP(t));
    ar(CEREAL_NVP(iterate));
    ar(CEREAL_NVP(rand_max));
    ar(CEREAL_NVP(epoch_size));
    ar(CEREAL_NVP(tol));
    ar(CEREAL_NVP(rand_type));
    ar(CEREAL_NVP(permutation));
    ar(CEREAL_NVP(i_perm));
    ar(CEREAL_NVP(permutation_ready));
    ar(CEREAL_NVP(record_every));
    // Note that only the seed is part of the serialization.
    // If the generator has been used (i.e. numbers have been drawn from it)
    // this will not be reflected in the restored (deserialized) object.
    const auto rand_seed = rand.get_seed();
    ar(CEREAL_NVP(rand_seed));
  }

  BoolStrReport compare(const TStoSolver<T, K> &that, std::stringstream &ss) {
    return BoolStrReport(TICK_CMP_REPORT(ss, t) && TICK_CMP_REPORT(ss, iterate) &&
                             TICK_CMP_REPORT(ss, rand_max) && TICK_CMP_REPORT(ss, epoch_size) &&
                             TICK_CMP_REPORT(ss, tol) && TICK_CMP_REPORT(ss, rand_type) &&
                             TICK_CMP_REPORT(ss, permutation) && TICK_CMP_REPORT(ss, i_perm) &&
                             TICK_CMP_REPORT(ss, permutation_ready) &&
                             TICK_CMP_REPORT(ss, record_every),
                         ss.str());
  }
};

template <typename T, typename K>
inline std::ostream &operator<<(std::ostream &s, const TStoSolver<T, K> &p) {
  return s << typeid(p).name();
}

using StoSolver = TStoSolver<double, double>;

using StoSolverDouble = TStoSolver<double, double>;
CEREAL_SPECIALIZE_FOR_ALL_ARCHIVES(StoSolverDouble, cereal::specialization::member_load_save)

using StoSolverFloat = TStoSolver<float, float>;
CEREAL_SPECIALIZE_FOR_ALL_ARCHIVES(StoSolverFloat, cereal::specialization::member_load_save)

using StoSolverAtomicDouble = TStoSolver<double, std::atomic<double> >;
CEREAL_SPECIALIZE_FOR_ALL_ARCHIVES(StoSolverAtomicDouble, cereal::specialization::member_load_save)

using StoSolverAtomicFloat = TStoSolver<float, std::atomic<float> >;
CEREAL_SPECIALIZE_FOR_ALL_ARCHIVES(StoSolverAtomicFloat, cereal::specialization::member_load_save)

#endif  // LIB_INCLUDE_TICK_SOLVER_STO_SOLVER_H_
