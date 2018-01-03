// License: BSD 3 clause


%{
#include "hawkes_custom.h"
%}


class Hawkes_custom : public Hawkes {
    public :
    Hawkes_custom::Hawkes_custom(unsigned int n_nodes, int seed, ulong _MaxN_of_f, const SArrayDoublePtrList1D &_f_i);
};

// laisse tomber le cereal
// TICK_MAKE_PICKLABLE(Hawkes_custom);