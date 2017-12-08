// License: BSD 3 clause

%include <std_shared_ptr.i>

%{
#include "tick/solver/svrg.h"
#include "tick/base_model/model.h"
%}

class SVRG : public StoSolver {
  public:
    enum class VarianceReductionMethod {
        Last    = 1,
        Average = 2,
        Random  = 3
    };

    enum class StepType {
        Fixed = 1,
        BarzilaiBorwein = 2,
    };

    SVRG(ulong epoch_size,
         double tol,
         RandType rand_type,
         double step,
         int seed = -1,
         int n_threads = 1,
         VarianceReductionMethod variance_reduction = VarianceReductionMethod::Last,
         StepType step_method = StepType::Fixed);

    void solve();

    double get_step();
    void set_step(double step);

    VarianceReductionMethod get_variance_reduction();
    void set_variance_reduction(VarianceReductionMethod variance_reduction);

    StepType get_step_type();
    void set_step_type(StepType step_type);
};
