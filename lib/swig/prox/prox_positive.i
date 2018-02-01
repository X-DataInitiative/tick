// License: BSD 3 clause

%{
#include "tick/prox/prox_positive.h"
%}

class ProxPositive : public TProx<double, double> {
 public:
   ProxPositive(double strength);

   ProxPositive(double strength,
                ulong start,
                ulong end);
};
