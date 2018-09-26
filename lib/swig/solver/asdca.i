// License: BSD 3 clause

%include "sto_solver.i"

%{
#include "tick/solver/asdca.h"
%}


template <class T>
class AtomicSDCA : public TStoSolver<T, std::atomic<T>> {
 public:
    AtomicSDCA();

    AtomicSDCA(
      T l_l2sq,
      ulong epoch_size = 0,
      T tol = 0.,
      RandType rand_type = RandType::unif,
      int record_every = 1,
      int seed = -1,
      int n_threads=2
    );

    void solve(int n_epochs = 1);
    void solve_batch(int n_epochs = 1, ulong bach_size = 2);

    void set_model(std::shared_ptr<TModel<T, std::atomic<T> > > model);
    void reset();
    void set_starting_iterate();
//    void set_starting_iterate(Array<std::atomic<T> > &dual_vector);
//    std::shared_ptr<Array<std::atomic<T> > > get_primal_vector();
//    std::shared_ptr<Array<std::atomic<T> > > get_dual_vector();
    void get_minimizer(Array<T> & out);

    T get_l_l2sq() const;
    void set_l_l2sq(T l_l2sq);
};

%template(AtomicSDCADouble) AtomicSDCA<double>;
typedef AtomicSDCA<double> AtomicSDCADouble;
//TICK_MAKE_TEMPLATED_PICKLABLE(AtomicSDCA, AtomicSDCADouble , double);

%template(AtomicSDCAFloat) AtomicSDCA<float>;
typedef AtomicSDCA<float> AtomicSDCAFloat;
//TICK_MAKE_TEMPLATED_PICKLABLE(AtomicSDCA, AtomicSDCAFloat , float);
