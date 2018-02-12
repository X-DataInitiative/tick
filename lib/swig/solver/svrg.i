// License: BSD 3 clause

%include <std_shared_ptr.i>

%{
#include "tick/solver/svrg.h"
#include "tick/base_model/model.h"
%}

template <class T>
class TSVRG : public TStoSolver<T> {
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

    TSVRG(ulong epoch_size,
         T tol,
         RandType rand_type,
         T step,
         int seed = -1,
         int n_threads = 1,
         VarianceReductionMethod variance_reduction = VarianceReductionMethod::Last,
         StepType step_method = StepType::Fixed);

    void solve();

    T get_step();
    void set_step(T step);

    VarianceReductionMethod get_variance_reduction();
    void set_variance_reduction(VarianceReductionMethod variance_reduction);

    StepType get_step_type();
    void set_step_type(StepType step_type);
};

%template(SVRG) TSVRG<double>; 
typedef TSVRG<double> SVRG;

%template(SVRGDouble) TSVRG<double>;
typedef TSVRG<double> SVRGDouble;

%template(SVRGFloat) TSVRG<float>;
typedef TSVRG<double> SVRGFloat;
