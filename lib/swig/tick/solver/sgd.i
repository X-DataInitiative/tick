// License: BSD 3 clause

%{
#include "tick/solver/sgd.h"
#include "tick/base_model/model.h"
%}

template <class T, class K = T>
class TSGD : public TStoSolver<T, K> {

public:
    TSGD();
    TSGD(size_t epoch_size,
        T tol,
        RandType rand_type,
        T step,
        size_t record_every = 1,
        int seed = -1);

    inline void set_step(T step);

    inline T get_step() const;

    bool compare(const TSGD<T, K> &that);
};

%template(SGDDouble) TSGD<double>;
typedef TSGD<double> SGDDouble;
TICK_MAKE_TEMPLATED_PICKLABLE(TSGD, SGDDouble , double);

%template(SGDFloat) TSGD<float>;
typedef TSGD<float> SGDFloat;
TICK_MAKE_TEMPLATED_PICKLABLE(TSGD, SGDFloat , float);
