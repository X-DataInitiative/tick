// License: BSD 3 clause
/*
%include "sto_solver.i"

%{
#include "tick/solver/asaga.h"
%}

template <class T>
class AtomicSAGA : public TStoSolver<T, T> {
 public:
    AtomicSAGA();

    AtomicSAGA(
      size_t epoch_size,
      T tol,
      RandType rand_type,
      T step,
      size_t record_every = 1,
      int seed = -1,
      int n_threads = 2
    );

    void solve(size_t n_epochs = 1);
};

%template(AtomicSAGADouble) AtomicSAGA<double>;
typedef AtomicSAGA<double> AtomicSAGADouble;

%template(AtomicSAGAFloat) AtomicSAGA<float>;
typedef AtomicSAGA<float> AtomicSAGAFloat;
