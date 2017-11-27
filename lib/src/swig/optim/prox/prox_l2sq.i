// License: BSD 3 clause

%{
#include "tick/optim/prox/prox_l2sq.h"
%}

class ProxL2Sq : public ProxSeparable {
 public:
   ProxL2Sq(double strength,
            bool positive);

   ProxL2Sq(double strength,
            ulong start,
            ulong end,
            bool positive);
};
