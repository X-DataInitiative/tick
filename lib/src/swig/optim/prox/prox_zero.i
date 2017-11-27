// License: BSD 3 clause

%{
#include "tick/optim/prox/prox_zero.h"
%}

class ProxZero : public Prox {
 public:
   ProxZero(double strength);

   ProxZero(double strength,
            unsigned long start,
            unsigned long end);
};
