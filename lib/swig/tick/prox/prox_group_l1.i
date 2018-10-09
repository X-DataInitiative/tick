// License: BSD 3 clause

%{
#include "tick/prox/prox_group_l1.h"
%}

template <class T, class K>
class TProxGroupL1 : public TProxWithGroups<T, K> {
 protected:
  std::unique_ptr<TProx<T, K> > build_prox(T strength, ulong start, ulong end, bool positive);

 public:
  TProxGroupL1(T strength, SArrayULongPtr blocks_start, SArrayULongPtr blocks_length,
                bool positive);

  TProxGroupL1(T strength, SArrayULongPtr blocks_start, SArrayULongPtr blocks_length,
                ulong start, ulong end, bool positive);

  inline void set_positive(bool positive);

  inline virtual void set_blocks_start(SArrayULongPtr blocks_start);

  inline virtual void set_blocks_length(SArrayULongPtr blocks_length);

  bool compare(const TProxGroupL1<T, K> &that);
};

%template(ProxGroupL1Double) TProxGroupL1<double, double>;
typedef TProxGroupL1<double, double> ProxGroupL1Double;

%template(ProxGroupL1Float) TProxGroupL1<float, float>;
typedef TProxGroupL1<float, float> ProxGroupL1Float;

