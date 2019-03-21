// License: BSD 3 clause

%{
#include "tick/prox/prox_binarsity.h"
%}

%include "prox_with_groups.i"

template <class T, class K>
class TProxBinarsity : public TProxWithGroups<T, K> {
 public:
  TProxBinarsity();
  TProxBinarsity(T strength, SArrayULongPtr blocks_start, SArrayULongPtr blocks_length,
                bool positive);

  TProxBinarsity(T strength, SArrayULongPtr blocks_start, SArrayULongPtr blocks_length,
                ulong start, ulong end, bool positive);

  inline void set_positive(bool positive);

  inline virtual void set_blocks_start(SArrayULongPtr blocks_start);

  inline virtual void set_blocks_length(SArrayULongPtr blocks_length);

  bool compare(const TProxBinarsity<T, K> &that);
};

%template(ProxBinarsityDouble) TProxBinarsity<double, double>;
typedef TProxBinarsity<double, double> ProxBinarsityDouble;
TICK_MAKE_TK_PICKLABLE(TProxBinarsity, ProxBinarsityDouble ,  double,  double);

%template(ProxBinarsityFloat) TProxBinarsity<float, float>;
typedef TProxBinarsity<float, float> ProxBinarsityFloat;
TICK_MAKE_TK_PICKLABLE(TProxBinarsity, ProxBinarsityFloat ,  float,  float);
