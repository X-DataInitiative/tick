%{
#include "prox_l2sq.h"
%}


class ProxL2Sq : public ProxSeparable {


public:

    ProxL2Sq(double strength, bool positive);

    ProxL2Sq(double strength, unsigned long start, unsigned long end,
             bool positive);

    inline virtual void set_positive(bool positive);
};
