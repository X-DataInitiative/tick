// License: BSD 3 clause

%{
#include "tick/prox/prox_equality.h"
%}

class ProxEquality : public TProx<double, double> {
 public:
   ProxEquality(double strength,
                bool positive);

   ProxEquality(double strength,
                ulong start,
                ulong end,
                bool positive);
};
