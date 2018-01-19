// License: BSD 3 clause


%include defs.i

%{
#include "tick/hawkes/simulation/simu_poisson_process.h"
%}

class Poisson : public PP {

    public :
        Poisson(double intensity, int seed = -1);
        Poisson(SArrayDoublePtr intensities, int seed = -1);
        virtual ~Poisson();
        SArrayDoublePtr get_intensities();
};