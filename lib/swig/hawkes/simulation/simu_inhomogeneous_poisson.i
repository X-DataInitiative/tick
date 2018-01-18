// License: BSD 3 clause

%{
#include "tick/hawkes/simulation/simu_inhomogeneous_poisson.h"
%}

%include "std_vector.i"
%template(TimeFunctionVector) std::vector<TimeFunction>;

class InhomogeneousPoisson : public PP {

    public :
        InhomogeneousPoisson(TimeFunction intensity_function, int seed = -1);
        InhomogeneousPoisson(std::vector<TimeFunction> intensity_functions, int seed = -1);
        virtual ~InhomogeneousPoisson();
        //TODO: handle it by returning TimeFunctions to Python...
        SArrayDoublePtr intensity_value(int dimension, ArrayDouble & times_values);
};