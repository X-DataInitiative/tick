%{
#include "prox_tv.h"
%}


class ProxTV : public Prox {


public:

    ProxTV(double strength,
           bool positive);

    ProxTV(double strength,
           unsigned long start,
           unsigned long end,
           bool positive);

    inline virtual void set_positive(bool positive);
};
