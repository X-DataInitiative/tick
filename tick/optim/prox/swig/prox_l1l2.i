%{
#include "prox_l1l2.h"
%}

class ProxL1L2 : public Prox {
 public:
  ProxL1L2(double strength, bool positive);

  ProxL1L2(double strength, ulong start, ulong end, bool positive);
};