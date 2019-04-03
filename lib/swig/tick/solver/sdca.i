// License: BSD 3 clause

%include "sto_solver.i"

%{
#include "tick/solver/sdca.h"
%}

template <class T>
class TSDCA : public TStoSolver<T, T> {
 public:
  TSDCA();
  TSDCA(T l_l2sq, ulong epoch_size, T tol,
        RandType rand_type, size_t record_every = 1, int seed = -1,
        int n_threads = 1);

  T get_l_l2sq();
  void set_l_l2sq(T l_l2sq);
  void set_step_size(T step_size);

  std::shared_ptr<SArray<T> > get_primal_vector();
  std::shared_ptr<SArray<T> > get_dual_vector();

  void solve_batch(size_t n_epochs = 1, ulong batch_size = 1);
  void set_starting_iterate(Array<T> &dual_vector);

  bool compare(const TSDCA<T> &that);
};

%template(SDCADouble) TSDCA<double>;
typedef TSDCA<double> SDCADouble;
TICK_MAKE_TEMPLATED_PICKLABLE(TSDCA, SDCADouble , double);

%template(SDCAFloat) TSDCA<float>;
typedef TSDCA<float> SDCAFloat;
TICK_MAKE_TEMPLATED_PICKLABLE(TSDCA, SDCAFloat , float);

// AtomicSDCA
template <class T>
class TAtomicSDCA : public TStoSolver<T, std::atomic<T> > {
 public:
  TAtomicSDCA();
  TAtomicSDCA(T l_l2sq, ulong epoch_size, T tol,
              RandType rand_type, size_t record_every = 1, int seed = -1,
              int n_threads=2);

  bool compare(const TAtomicSDCA<T> &that);
};

%template(AtomicSDCADouble) TAtomicSDCA<double>;
typedef TAtomicSDCA<double> AtomicSDCADouble;

%template(AtomicSDCAFloat) TAtomicSDCA<float>;
typedef TAtomicSDCA<float> AtomicSDCAFloat;
