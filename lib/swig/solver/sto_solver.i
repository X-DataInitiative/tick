// License: BSD 3 clause

%include <std_shared_ptr.i>

%{
#include "tick/solver/sto_solver.h"
#include "tick/base_model/model.h"
%}

%include "array_module.i"
%include "model.i"
%include "prox.i"

// Type of randomness used when sampling at random data points
enum class RandType {
    unif = 0,
    perm
};

template <class T, class K>
class TStoSolver {
 public:
  TStoSolver(
    unsigned long epoch_size,
    double tol,
    RandType rand_type
  );
  virtual void solve();

  virtual void get_minimizer(Array<K> &out);
  virtual void get_iterate(Array<K> &out);
  virtual void set_starting_iterate(Array<K> &new_iterate);

  inline void set_tol(double tol);
  inline double get_tol() const;
  inline void set_epoch_size(unsigned long epoch_size);
  inline unsigned long get_epoch_size() const;
  inline void set_rand_type(RandType rand_type);
  inline RandType get_rand_type() const;
  inline void set_rand_max(unsigned long rand_max);
  inline unsigned long get_rand_max() const;

  virtual void set_model(std::shared_ptr<TModel<T, K> > model);
  virtual void set_prox(std::shared_ptr<TProx<T, K> > prox);
  void set_seed(int seed);
};

%rename(TStoSolverDouble) TStoSolver<double, double>; 
class TStoSolver<double, double> {
 // Base abstract for a stochastic solver
 public:
  TStoSolverDouble(
    unsigned long epoch_size,
    double tol,
    RandType rand_type
  );

  virtual void solve();

  virtual void get_minimizer(ArrayDouble &out);
  virtual void get_iterate(ArrayDouble &out);
  virtual void set_starting_iterate(ArrayDouble &new_iterate);

  inline void set_tol(double tol);
  inline double get_tol() const;
  inline void set_epoch_size(unsigned long epoch_size);
  inline unsigned long get_epoch_size() const;
  inline void set_rand_type(RandType rand_type);
  inline RandType get_rand_type() const;
  inline void set_rand_max(unsigned long rand_max);
  inline unsigned long get_rand_max() const;

  virtual void set_model(std::shared_ptr<TModel<double, double> > model);
  virtual void set_prox(std::shared_ptr<TProx<double, double> > prox);
  void set_seed(int seed);
};

%rename(TStoSolverFloat) TStoSolver<float, float>; 
class TStoSolver<float, float> {
 // Base abstract for a stochastic solver
 public:
  TStoSolverFloat(
    unsigned long epoch_size,
    float tol,
    RandType rand_type
  );

  virtual void solve();

  virtual void get_minimizer(ArrayFloat &out);
  virtual void get_iterate(ArrayFloat &out);
  virtual void set_starting_iterate(ArrayFloat &new_iterate);

  inline void set_tol(float tol);
  inline float get_tol() const;
  inline void set_epoch_size(unsigned long epoch_size);
  inline unsigned long get_epoch_size() const;
  inline void set_rand_type(RandType rand_type);
  inline RandType get_rand_type() const;
  inline void set_rand_max(unsigned long rand_max);
  inline unsigned long get_rand_max() const;

  virtual void set_model(ModelFloatPtr model);
  virtual void set_prox(std::shared_ptr<TProx<float, float> > prox);
  void set_seed(int seed);
};

class StoSolver : public TStoSolver<double, double> {
 // Base abstract for a stochastic solver
 public:
  StoSolver(
    unsigned long epoch_size,
    double tol,
    RandType rand_type
  );

  virtual void solve();
  virtual void set_model(std::shared_ptr<TModel<double, double> > model);
  virtual void set_prox(std::shared_ptr<TProx<double, double> > prox);

  virtual void get_iterate(ArrayDouble &out);
  virtual void set_starting_iterate(ArrayDouble &new_iterate);
  virtual void get_minimizer(ArrayDouble &out);
};
