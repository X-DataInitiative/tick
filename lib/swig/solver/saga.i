// License: BSD 3 clause

%include "sto_solver.i"

%{
#include "tick/solver/saga.h"
%}

template <class T>
class TSAGA : public TStoSolver<T, T> {
 public:
    TSAGA();
    TSAGA(unsigned long epoch_size,
         T tol,
         RandType rand_type,
         T step,
         int seed);
    void solve();
    void set_step(T step);

    void set_model(std::shared_ptr<TModel<T, T> > model) override;

    bool compare(const TSAGA<T> &that);
};

%template(SAGADouble) TSAGA<double>;
typedef TSAGA<double> SAGADouble;
TICK_MAKE_TEMPLATED_PICKLABLE(TSAGA, SAGADouble, double);

%template(SAGAFloat) TSAGA<float>;
typedef TSAGA<float> SAGAFloat;
TICK_MAKE_TEMPLATED_PICKLABLE(TSAGA, SAGAFloat , float);
