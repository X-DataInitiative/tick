// License: BSD 3 clause

%include <std_shared_ptr.i>

%{
#include "tick/solver/sgd.h"
#include "tick/base_model/model.h"
%}

template <class T>
class TSGD : public TStoSolver<T> {

public:

    TSGD(unsigned long epoch_size,
        T tol,
        RandType rand_type,
        T step,
        int seed);

    inline void set_step(T step);

    inline T get_step() const;

    void solve();
};

%template(SGD) TSGD<double>; 
typedef TSGD<double> SGD;

%template(SGDDouble) TSGD<double>;
typedef TSGD<double> SGDDouble;

%template(SGDFloat) TSGD<float>;
typedef TSGD<double> SGDFloat;
