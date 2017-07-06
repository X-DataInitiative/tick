// License: BSD 3 clause

%include <std_shared_ptr.i>

%{
#include "adagrad.h"
#include "model.h"
%}

class AdaGrad : public StoSolver {

public:

    AdaGrad(unsigned long epoch_size,
        double tol,
        RandType rand_type,
        double step,
        int seed);

    void solve();
};
