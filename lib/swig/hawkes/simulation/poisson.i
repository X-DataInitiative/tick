// License: BSD 3 clause


%include defs.i

%{
#include "tick/hawkes/simulation/poisson.h"
%}

class Poisson : public PP {

    public :
        Poisson(double intensity, int seed = -1);
        Poisson(SArrayDoublePtr intensities, int seed = -1);
        virtual ~Poisson();
        SArrayDoublePtr get_intensities();
};