// License: BSD 3 clause

%{
#include "tick/optim/prox/prox_tv.h"
%}

class ProxTV : public Prox {
 public:
   ProxTV(double strength,
          bool positive);

   ProxTV(double strength,
          unsigned long start,
          unsigned long end,
          bool positive);
};
