// License: BSD 3 clause


%{
#include "hawkes_custom.h"
%}


class Hawkes_custom : public Hawkes {
    public :
    Hawkes_custom::Hawkes_custom(unsigned int n_nodes, int seed, ulong _MaxN_of_f, const SArrayDoublePtrList1D &_f_i);
    Hawkes_custom::Hawkes_custom(unsigned int n_nodes, int seed, ulong _MaxN_of_f, const SArrayDoublePtrList1D &_f_i, const ArrayDouble &extrainfo, const std::string _simu_mode);
    VArrayULongPtr get_global_n();
    VArrayDoublePtr get_Qty();
};

// laisse tomber le cereal
// TICK_MAKE_PICKLABLE(Hawkes_custom);