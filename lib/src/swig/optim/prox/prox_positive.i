// License: BSD 3 clause

%{
#include "tick/optim/prox/prox_positive.h"
%}

class ProxPositive : public Prox {
 public:
   ProxPositive(double strength);

   ProxPositive(double strength,
                ulong start,
                ulong end);
};
