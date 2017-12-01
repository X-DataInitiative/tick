// License: BSD 3 clause

%{
#include "tick/prox/prox_l2.h"
%}

class ProxL2 : public Prox {
 public:
   ProxL2(double strength,
          bool positive);

   ProxL2(double strength,
          unsigned long start,
          unsigned long end,
          bool positive);
};
