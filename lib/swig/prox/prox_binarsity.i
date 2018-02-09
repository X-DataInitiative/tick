// License: BSD 3 clause

%{
#include "tick/prox/prox_binarsity.h"
%}

%include "prox_with_groups.i"

template <class T>
class TProxBinarsity : public TProxWithGroups<T> {
 public:
  TProxBinarsity(T strength, SArrayULongPtr blocks_start, SArrayULongPtr blocks_length,
                bool positive);

  TProxBinarsity(T strength, SArrayULongPtr blocks_start, SArrayULongPtr blocks_length,
                ulong start, ulong end, bool positive);

  inline void set_positive(bool positive);

  inline virtual void set_blocks_start(SArrayULongPtr blocks_start);

  inline virtual void set_blocks_length(SArrayULongPtr blocks_length);
};

%template(ProxBinarsity) TProxBinarsity<double>;
typedef TProxBinarsity<double> ProxBinarsity;

%template(ProxBinarsityDouble) TProxBinarsity<double>;
typedef TProxBinarsity<double> ProxBinarsityDouble;

%template(ProxBinarsityFloat) TProxBinarsity<float>;
typedef TProxBinarsity<float> ProxBinarsityFloat;
