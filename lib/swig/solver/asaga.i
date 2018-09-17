// License: BSD 3 clause

%include "sto_solver.i"

%{
#include "tick/solver/asaga.h"
%}

template <class T>
class AtomicSAGA : public TStoSolver<T, T> {
 public:
    AtomicSAGA();

    AtomicSAGA(unsigned long epoch_size,
               unsigned long iterations,
               T tol, RandType rand_type,
               T step, int seed, int n_threads = 2
               );

    void solve();
    void set_step(T step);

    void set_model(std::shared_ptr<TModel<T, T> > model) override;
};

%template(AtomicSAGADouble) AtomicSAGA<double>;
typedef AtomicSAGA<double> AtomicSAGADouble;

%template(AtomicSAGAFloat) AtomicSAGA<float>;
typedef AtomicSAGA<float> AtomicSAGAFloat;
