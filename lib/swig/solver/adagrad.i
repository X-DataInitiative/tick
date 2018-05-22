// License: BSD 3 clause

%include <std_shared_ptr.i>

%{
#include "tick/solver/adagrad.h"
#include "tick/base_model/model.h"
%}

template <class T, class K = T>
class TAdaGrad : public TStoSolver<T, K> {
public:
    TAdaGrad();
    TAdaGrad(unsigned long epoch_size,
        T tol,
        RandType rand_type,
        T step,
        int seed);

    void solve();

    bool compare(const TAdaGrad<T, K> &that);
};

%template(AdaGradDouble) TAdaGrad<double>;
typedef TAdaGrad<double> AdaGradDouble;
TICK_MAKE_TEMPLATED_PICKLABLE(TAdaGrad, AdaGradDouble, double);

%template(AdaGradFloat) TAdaGrad<float>;
typedef TAdaGrad<float> AdaGradFloat;
TICK_MAKE_TEMPLATED_PICKLABLE(TAdaGrad, AdaGradFloat , float);
