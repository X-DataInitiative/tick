// License: BSD 3 clause

%include <std_shared_ptr.i>

%{
#include "tick/solver/saga.h"
#include "tick/base_model/model.h"
%}

class SAGA : public StoSolver {

public:
    enum class VarianceReductionMethod {
        Last    = 1,
        Average = 2,
        Random  = 3
    };

    SAGA(unsigned long epoch_size,
         double tol,
         RandType rand_type,
         double step,
         int seed,
         VarianceReductionMethod variance_reduction = VarianceReductionMethod::Last);

    void solve();

    void set_step(double step);

    VarianceReductionMethod get_variance_reduction();

    void set_variance_reduction(VarianceReductionMethod variance_reduction);
};
