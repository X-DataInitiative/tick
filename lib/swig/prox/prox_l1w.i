// License: BSD 3 clause

%{
#include "tick/prox/prox_l1w.h"
%}

class ProxL1w : public TProx<double, double> {
 public:
   ProxL1w(double strength,
           SArrayDoublePtr weights,
           bool positive);

   ProxL1w(double strength,
           SArrayDoublePtr weights,
           ulong start,
           ulong end,
           bool positive);

   virtual void set_weights(SArrayDoublePtr weights) final;
};
