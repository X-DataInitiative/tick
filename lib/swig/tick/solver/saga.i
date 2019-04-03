// License: BSD 3 clause

%include "sto_solver.i"

%{
#include "tick/solver/saga.h"
%}

template <class T, class K>
class TSAGA : public TStoSolver<T, K> {
 public:
    TSAGA();

    TSAGA(size_t epoch_size,
         T tol,
         RandType rand_type,
         T step,
         size_t record_every = 1,
         int seed = -1,
         int n_threads = 1);
    void set_step(T step);

    bool compare(const TSAGA<T, K> &that);
};

%template(SAGADouble) TSAGA<double, double>;
typedef TSAGA<double, double> SAGADouble;
TICK_MAKE_TEMPLATED_PICKLABLE(TSAGA, SAGADouble, %arg(double, double));

%template(SAGAFloat) TSAGA<float, float>;
typedef TSAGA<float, float> SAGAFloat;
TICK_MAKE_TEMPLATED_PICKLABLE(TSAGA, SAGAFloat, %arg(float, float));


%template(SAGADoubleAtomicIterate) TSAGA<double, std::atomic<double> >;
typedef TSAGA<double, std::atomic<double> > SAGADoubleAtomicIterate;


template <class T, class K, std::memory_order M=std::memory_order_seq_cst,
           std::memory_order N=std::memory_order_seq_cst, bool O=true>
class AtomicSAGA : public TStoSolver<T, K> {
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
};

%template(AtomicSAGAFloat) AtomicSAGA<float, float>;
typedef AtomicSAGA<float, float> AtomicSAGAFloat;

%template(AtomicSAGADoubleAtomicIterate) AtomicSAGA<double, std::atomic<double> >;
typedef AtomicSAGA<double, std::atomic<double> > AtomicSAGADoubleAtomicIterate;


%template(AtomicSAGADouble) AtomicSAGA<double, double>;
typedef AtomicSAGA<double, double> AtomicSAGADouble;

%template(AtomicSAGARelax) AtomicSAGA<double, double, std::memory_order_relaxed, std::memory_order_relaxed>;
typedef AtomicSAGA<double, double, std::memory_order_relaxed, std::memory_order_relaxed> AtomicSAGARelax;

%template(AtomicSAGANoLoad) AtomicSAGA<double, double, std::memory_order_seq_cst, std::memory_order_seq_cst, false>;
typedef AtomicSAGA<double, double, std::memory_order_seq_cst, std::memory_order_seq_cst, false> AtomicSAGANoLoad;

template <class T, std::memory_order M=std::memory_order_seq_cst,
           std::memory_order N=std::memory_order_seq_cst>
class ExtraAtomicSAGA : public TStoSolver<T, std::atomic<T>> {
 public:
    ExtraAtomicSAGA();

    ExtraAtomicSAGA(
      size_t epoch_size,
      T tol,
      RandType rand_type,
      T step,
      size_t record_every = 1,
      int seed = -1,
      int n_threads = 2
    );
};

%template(ExtraAtomicSAGADouble) ExtraAtomicSAGA<double>;
typedef ExtraAtomicSAGA<double> ExtraAtomicSAGADouble;

%template(ExtraAtomicSAGADoubleRelax) ExtraAtomicSAGA<double, std::memory_order_relaxed, std::memory_order_relaxed>;
typedef ExtraAtomicSAGA<double, std::memory_order_relaxed, std::memory_order_relaxed> ExtraAtomicSAGADoubleRelax;
