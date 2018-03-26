// License: BSD 3 clause

%include <std_shared_ptr.i>

%{
#include "tick/solver/svrg.h"
%}

enum class SVRG_VarianceReductionMethod : uint16_t {
  Last = 1,
  Average = 2,
  Random = 3,
};
enum class SVRG_StepType : uint16_t {
    Fixed = 1,
    BarzilaiBorwein = 2,
};

template <class T>
class TSVRG : public TStoSolver<T> {
  public:
    TSVRG();
    TSVRG(ulong epoch_size,
         T tol,
         RandType rand_type,
         T step,
         int seed = -1,
         int n_threads = 1,
         SVRG_VarianceReductionMethod variance_reduction = SVRG_VarianceReductionMethod::Last,
         SVRG_StepType step_method = SVRG_StepType::Fixed);

    void solve();

    T get_step();
    void set_step(T step);

    SVRG_VarianceReductionMethod get_variance_reduction();
    void set_variance_reduction(SVRG_VarianceReductionMethod variance_reduction);

    SVRG_StepType get_step_type();
    void set_step_type(SVRG_StepType step_type);

    bool compare(const TSVRG<T> &that);
};

%template(SVRGDouble) TSVRG<double>;
typedef TSVRG<double> SVRGDouble;
TICK_MAKE_TEMPLATED_PICKLABLE(TSVRG, SVRGDouble, double);

%template(SVRGFloat) TSVRG<float>;
typedef TSVRG<float> SVRGFloat;
TICK_MAKE_TEMPLATED_PICKLABLE(TSVRG, SVRGFloat , float);
