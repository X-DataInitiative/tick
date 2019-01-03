// License: BSD 3 clause

%include <std_shared_ptr.i>

%include "sto_solver.i"

%{
#include "tick/solver/sdca.h"
%}

template <class T, class K>
class TBaseSDCA : public TStoSolver<T, K> {
 public:
  TBaseSDCA();
  TBaseSDCA(T l_l2sq, ulong epoch_size, T tol,
            RandType rand_type, int record_every = 1, int seed = -1,
            int n_threads = 1);

  T get_l_l2sq();
  void set_l_l2sq(T l_l2sq);
  void set_step_size(T step_size);

  std::shared_ptr<SArray<T> > get_primal_vector();
  std::shared_ptr<SArray<T> > get_dual_vector();

  void solve_batch(int n_epochs = 1, ulong batch_size = 1);
  void set_starting_iterate(Array<T> &dual_vector);
};

// INSTANTIATIONS

%rename(BaseSDCADoubleDouble) TBaseSDCA<double, double>;
class TBaseSDCA<double, double> : public TStoSolver<double, double> {
 // Base abstract for a stochastic solver
 public:
    TBaseSDCA();
    TBaseSDCA(double l_l2sq, ulong epoch_size, double tol,
              RandType rand_type, int record_every = 1, int seed = -1,
              int n_threads = 1);
  
    double get_l_l2sq();
    void set_l_l2sq(double l_l2sq);
    void set_step_size(double step_size);
  
    SArrayDoublePtr get_primal_vector();
    SArrayDoublePtr get_dual_vector();
  
    void solve_batch(int n_epochs = 1, ulong batch_size = 1);
    void set_starting_iterate(Array<double> &dual_vector);
};
typedef TBaseSDCA<double, double> BaseSDCADoubleDouble;

%rename(BaseSDCAFloatFloat) TBaseSDCA<float, float>;
class TBaseSDCA<float, float> : public TStoSolver<float, float> {
 // Base abstract for a stochastic solver
 public:
    TBaseSDCA();
    TBaseSDCA(float l_l2sq, ulong epoch_size, float tol,
              RandType rand_type, int record_every = 1, int seed = -1,
              int n_threads = 1);
  
    float get_l_l2sq();
    void set_l_l2sq(float l_l2sq);
    void set_step_size(float step_size);
  
    SArrayFloatPtr get_primal_vector();
    SArrayFloatPtr get_dual_vector();
  
    void solve_batch(int n_epochs = 1, ulong batch_size = 1);
    void set_starting_iterate(Array<float> &dual_vector);
};
typedef TBaseSDCA<float, float> BaseSDCAFloatFloat;


%rename(BaseSDCADoubleAtomicDouble) TBaseSDCA<double, std::atomic<double> >;
class TBaseSDCA<double, std::atomic<double> > : public TStoSolver<double, std::atomic<double> > {
 // Base abstract for a stochastic solver
 public:
    TBaseSDCA();
    TBaseSDCA(double l_l2sq, ulong epoch_size, double tol,
              RandType rand_type, int record_every = 1, int seed = -1,
              int n_threads = 1);
  
    double get_l_l2sq();
    void set_l_l2sq(double l_l2sq);
    void set_step_size(double step_size);
  
    SArrayDoublePtr get_primal_vector();
    SArrayDoublePtr get_dual_vector();
  
    void solve_batch(int n_epochs = 1, ulong batch_size = 1);
    void set_starting_iterate(Array<double> &dual_vector);
};
typedef TBaseSDCA<double, std::atomic<double> > BaseSDCADoubleAtomicDouble;


%rename(BaseSDCAFloatAtomicFloat) TBaseSDCA<float, std::atomic<float> >;
class TBaseSDCA<float, std::atomic<float> > : public TStoSolver<float, std::atomic<float> > {
 // Base abstract for a stochastic solver
 public:
    TBaseSDCA();
    TBaseSDCA(float l_l2sq, ulong epoch_size, float tol,
              RandType rand_type, int record_every = 1, int seed = -1,
              int n_threads = 1);
  
    float get_l_l2sq();
    void set_l_l2sq(float l_l2sq);
    void set_step_size(float step_size);
  
    SArrayFloatPtr get_primal_vector();
    SArrayFloatPtr get_dual_vector();
  
    void solve_batch(int n_epochs = 1, ulong batch_size = 1);
    void set_starting_iterate(Array<float> &dual_vector);
};
typedef TBaseSDCA<float, std::atomic<float> > BaseSDCAFloatAtomicFloat;

// SDCA

template <class T>
class TSDCA : public TBaseSDCA<T, T> {

 public:
  TSDCA();
  TSDCA(T l_l2sq, ulong epoch_size, T tol,
        RandType rand_type, int record_every = 1, int seed = -1);
};

%template(SDCADouble) TSDCA<double>;
typedef TSDCA<double> SDCADouble;
//TICK_MAKE_PICKLABLE(SDCADouble);

%template(SDCAFloat) TSDCA<float>;
typedef TSDCA<float> SDCAFloat;
//TICK_MAKE_PICKLABLE(SDCAFloat);


// AtomicSDCA

template <class T>
 class TAtomicSDCA : public TBaseSDCA<T, std::atomic<T> > {

 public:
  TAtomicSDCA();
  TAtomicSDCA(T l_l2sq, ulong epoch_size, T tol,
              RandType rand_type, int record_every = 1, int seed = -1,
              int n_threads=2);
};

%template(AtomicSDCADouble) TAtomicSDCA<double>;
typedef TAtomicSDCA<double> AtomicSDCADouble;
//TICK_MAKE_PICKLABLE(SDCADouble);

%template(AtomicSDCAFloat) TAtomicSDCA<float>;
typedef TAtomicSDCA<float> AtomicSDCAFloat;
//TICK_MAKE_PICKLABLE(SDCAFloat);
