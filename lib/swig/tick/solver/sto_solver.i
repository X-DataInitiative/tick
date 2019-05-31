// License: BSD 3 clause

%include <std_shared_ptr.i>
%include "std_vector.i"

%{
#include "tick/solver/sto_solver.h"
#include "tick/base_model/model.h"
%}

%include "tick/array/array_module.i"
%include "tick/base_model/base_model_module.i"
%include "tick/prox/prox_module.i"

%template(IntVector) std::vector<int>;
%template(DoubleVector) std::vector<double>;
%template(SArrayDoubleVector) std::vector<std::shared_ptr<SArray<double> > >;
%template(SArrayFloatVector) std::vector<std::shared_ptr<SArray<float> > >;

// Type of randomness used when sampling at random data points
enum class RandType {
    unif = 0,
    perm
};

template <class T, class K = T>
class TStoSolver {
 public:
  TStoSolver(
    unsigned long epoch_size,
    T tol,
    RandType rand_type,
    int record_every = 1,
    int seed = -1
  );
  virtual void solve(int n_epochs = 1);

  virtual void get_minimizer(Array<T> &out);
  virtual void get_iterate(Array<T> &out);
  virtual void set_starting_iterate(Array<K> &new_iterate);

  inline void set_tol(T tol);
  inline T get_tol() const;
  inline void set_epoch_size(unsigned long epoch_size);
  inline unsigned long get_epoch_size() const;
  inline void set_rand_type(RandType rand_type);
  inline RandType get_rand_type() const;
  inline void set_rand_max(unsigned long rand_max);
  inline unsigned long get_rand_max() const;
  inline int get_record_every() const;
  inline void set_record_every(int record_every);

  std::vector<double> get_time_history() const;
  std::vector<int> get_epoch_history() const;
  std::vector<double> get_objectives() const;
  void set_prev_obj(const double obj);
  std::vector<std::shared_ptr<SArray<T> > > get_iterate_history() const;

  virtual void set_model(std::shared_ptr<TModel<T, K> > model);
  virtual void set_prox(std::shared_ptr<TProx<T, K> > prox);
  void set_seed(int seed);
  void reset();
};

%rename(StoSolverDouble) TStoSolver<double, double>;
class TStoSolver<double, double> {
 // Base abstract for a stochastic solver
 public:
  StoSolverDouble(
    unsigned long epoch_size,
    double tol,
    RandType rand_type
  );

  virtual void solve(int n_epochs = 1);
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
  inline int get_record_every() const;
  inline void set_record_every(int record_every);

  std::vector<double> get_time_history() const;
  std::vector<int> get_epoch_history() const;
  std::vector<double> get_objectives() const;
  void set_prev_obj(const double obj);
  SArrayDoublePtrList1D get_iterate_history() const;

  virtual void set_model(ModelDoublePtr model);
  virtual void set_prox(ProxDoublePtr prox);
  void set_seed(int seed);
  void reset();
};
typedef TStoSolver<double, double> StoSolverDouble;

%rename(StoSolverFloat) TStoSolver<float, float>;
class TStoSolver<float, float> {
 // Base abstract for a stochastic solver
 public:
  StoSolverFloat(
    unsigned long epoch_size,
    float tol,
    RandType rand_type
  );

  virtual void solve(int n_epochs = 1);

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
  inline int get_record_every() const;
  inline void set_record_every(int record_every);

  std::vector<double> get_time_history() const;
  std::vector<int> get_epoch_history() const;
  std::vector<double> get_objectives() const;
  void set_prev_obj(const double obj);
  SArrayFloatPtrList1D get_iterate_history() const;

  virtual void set_model(ModelFloatPtr model);
  virtual void set_prox(ProxFloatPtr prox);
  void set_seed(int seed);
  void reset();
};
typedef TStoSolver<float, float> StoSolverFloat;
