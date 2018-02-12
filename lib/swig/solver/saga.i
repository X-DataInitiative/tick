// License: BSD 3 clause

%include "sto_solver.i"

%{
#include "tick/solver/saga.h"
#include "tick/base_model/model.h"
%}

template <class T>
class TSAGA : public TStoSolver<T> {
 public:
    enum class VarianceReductionMethod {
        Last    = 1,
        Average = 2,
        Random  = 3
    };
    TSAGA(unsigned long epoch_size,
         T tol,
         RandType rand_type,
         T step,
         int seed,
         TSAGA::VarianceReductionMethod variance_reduction
         = TSAGA::VarianceReductionMethod::Last);
    void solve();
    void set_step(T step);
    VarianceReductionMethod get_variance_reduction();
    void set_variance_reduction(VarianceReductionMethod variance_reduction);

    void set_model(std::shared_ptr<TModel<T> > model) override;
};

%template(SAGA) TSAGA<double>; 
typedef TSAGA<double> SAGA;

%template(SAGADouble) TSAGA<double>;
typedef TSAGA<double> SAGADouble;

%template(SAGAFloat) TSAGA<float>;
typedef TSAGA<double> SAGAFloat;
