// License: BSD 3 clause

%include <std_shared_ptr.i>

%{
#include "tick/solver/adagrad.h"
#include "tick/base_model/model.h"
%}

template <class T>
class TAdaGrad : public TStoSolver<T> {
public:
    TAdaGrad(unsigned long epoch_size,
        T tol,
        RandType rand_type,
        T step,
        int seed);

    void solve();
};

%template(AdaGrad) TAdaGrad<double>; 
typedef TAdaGrad<double> AdaGrad;

%template(AdaGradDouble) TAdaGrad<double>;
typedef TAdaGrad<double> AdaGradDouble;

%template(AdaGradFloat) TAdaGrad<float>;
typedef TAdaGrad<double> AdaGradFloat;
