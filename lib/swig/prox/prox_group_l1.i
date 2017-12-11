// License: BSD 3 clause

%{
#include "tick/prox/prox_group_l1.h"
%}


class ProxGroupL1: public Prox {
 public:
  ProxGroupL1(double strength, SArrayULongPtr blocks_start,
                SArrayULongPtr blocks_length, bool positive);

  ProxGroupL1(double strength, SArrayULongPtr blocks_start,
                SArrayULongPtr blocks_length, ulong start,
                ulong end, bool positive);

  inline void set_positive(bool positive);

  inline virtual void set_blocks_start(SArrayULongPtr blocks_start);

  inline virtual void set_blocks_length(SArrayULongPtr blocks_length);
};

