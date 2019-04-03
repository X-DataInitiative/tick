// License: BSD 3 clause

%include std_shared_ptr.i

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

template <class T, class K = T>
class TSVRG : public TStoSolver<T, K> {
  public:
    TSVRG();
    TSVRG(ulong epoch_size,
         T tol,
         RandType rand_type,
         T step,
         size_t record_every = 1,
         int seed = -1,
         size_t n_threads = 1,
         SVRG_VarianceReductionMethod variance_reduction = SVRG_VarianceReductionMethod::Last,
         SVRG_StepType step_method = SVRG_StepType::Fixed);

    T get_step();
    void set_step(T step);

    SVRG_VarianceReductionMethod get_variance_reduction();
    void set_variance_reduction(SVRG_VarianceReductionMethod variance_reduction);

    SVRG_StepType get_step_type();
    void set_step_type(SVRG_StepType step_type);

    bool compare(const TSVRG<T, K> &that);
};

%template(SVRGDouble) TSVRG<double, double>;
typedef TSVRG<double, double> SVRGDouble;
TICK_MAKE_TK_PICKLABLE(TSVRG, SVRGDouble , double, double);

%template(SVRGFloat) TSVRG<float, float>;
typedef TSVRG<float, float> SVRGFloat;
TICK_MAKE_TK_PICKLABLE(TSVRG, SVRGFloat , float, float);

%template(SVRGDoubleAtomicIterate) TSVRG<double, std::atomic<double> >;
typedef TSVRG<double, std::atomic<double>> SVRGDoubleAtomicIterate;