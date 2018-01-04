// License: BSD 3 clause

%include <std_shared_ptr.i>

%{
#include "tick/optim/solver/saga.h"
#include "tick/optim/model/model.h"
%}

class SAGA : public StoSolver {

public:
    enum class VarianceReductionMethod {
        Last    = 1,
        Average = 2,
        Random  = 3
    };

    // Empty constructor only used for serialization
    SAGA(){};

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

TICK_MAKE_PICKLABLE(SAGA);
