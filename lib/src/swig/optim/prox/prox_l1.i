// License: BSD 3 clause

%{
#include "tick/optim/prox/prox_l1.h"
%}

class ProxL1 : public Prox {
 public:
   ProxL1(double strength = 0., bool positive = false);

   ProxL1(double strength,
          ulong start,
          ulong end,
          bool positive);
};

TICK_MAKE_PICKLABLE(ProxL1);