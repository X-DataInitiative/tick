// License: BSD 3 clause

%{
#include "tick/prox/prox_l1.h"
%}

class ProxL1 : public Prox {
 public:
   ProxL1(double strength,
          bool positive);

   ProxL1(double strength,
          ulong start,
          ulong end,
          bool positive);
};
