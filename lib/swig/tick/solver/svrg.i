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

template <class T, class K = T>
class TSVRG : public TStoSolver<T, K> {
  public:
    TSVRG();
    TSVRG(size_t epoch_size,
         T tol,
         RandType rand_type,
         T step,
         size_t record_every = 1,
         int seed = -1,
         size_t n_threads = 1,
         SVRG_VarianceReductionMethod variance_reduction = SVRG_VarianceReductionMethod::Last,
         SVRG_StepType step_method = SVRG_StepType::Fixed);
    void solve(size_t n_epochs = 1) override;

    T get_step();
    void set_step(T step);

    SVRG_VarianceReductionMethod get_variance_reduction();
    void set_variance_reduction(SVRG_VarianceReductionMethod variance_reduction);

    SVRG_StepType get_step_type();
    void set_step_type(SVRG_StepType step_type);

    bool compare(const TSVRG<T, K> &that);
    void set_prev_obj(const double obj);
};

%template(SVRGDouble) TSVRG<double, double>;
typedef TSVRG<double, double> SVRGDouble;
TICK_MAKE_TK_PICKLABLE(TSVRG, SVRGDouble , double, double);

%template(SVRGFloat) TSVRG<float, float>;
typedef TSVRG<float, float> SVRGFloat;
TICK_MAKE_TK_PICKLABLE(TSVRG, SVRGFloat , float, float);

%include std_vector.i

%template(SVRGDoublePtrVector) std::vector<SVRGDouble*>;
typedef std::vector<SVRGDouble*> SVRGDoublePtrVector;

template <typename T, typename K>
class MultiSVRG{
 public:
  static void multi_solve(std::vector<TSVRG<T, K>*> & solvers, size_t n_threads);
  static void push_solver(std::vector<TSVRG<T, K>*> & solvers, TSVRG<T, K> & solver);
};

%template(MultiSVRGDouble) MultiSVRG<double, double>;
typedef MultiSVRG<double, double> MultiSVRGDouble;

%template(MultiSVRGFloat) MultiSVRG<float, float>;
typedef MultiSVRG<float, float> MultiSVRGFloat;
