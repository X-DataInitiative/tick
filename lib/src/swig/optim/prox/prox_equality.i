// License: BSD 3 clause

%{
#include "tick/optim/prox/prox_equality.h"
%}

class ProxEquality : public Prox {
 public:
   ProxEquality(double strength,
                bool positive);

   ProxEquality(double strength,
                ulong start,
                ulong end,
                bool positive);
};
