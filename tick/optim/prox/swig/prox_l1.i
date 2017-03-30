%{
#include "prox_l1.h"
%}


class ProxL1 : public Prox {


public:

    ProxL1(double strength, bool positive);

    ProxL1(double strength, unsigned long start, unsigned long end,
           bool positive);

    inline virtual void set_positive(bool positive);
};
