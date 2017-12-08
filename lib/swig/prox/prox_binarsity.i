// License: BSD 3 clause

%{
#include "tick/prox/prox_binarsity.h"
%}


class ProxBinarsity: public Prox {

 public:

  ProxBinarsity(double strength, SArrayULongPtr blocks_start,
                SArrayULongPtr blocks_length, bool positive);

  ProxBinarsity(double strength, SArrayULongPtr blocks_start,
                SArrayULongPtr blocks_length, ulong start,
                ulong end, bool positive);

  inline void set_positive(bool positive);

  inline virtual void set_blocks_start(SArrayULongPtr blocks_start);

  inline virtual void set_blocks_length(SArrayULongPtr blocks_length);

};
