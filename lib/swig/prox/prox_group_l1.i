// License: BSD 3 clause

%{
#include "tick/prox/prox_group_l1.h"
%}

template <class T>
class TProxGroupL1 : public TProxWithGroups<T> {
 protected:
  std::unique_ptr<TProx<T> > build_prox(T strength, ulong start, ulong end, bool positive);

 public:
  TProxGroupL1(T strength, SArrayULongPtr blocks_start, SArrayULongPtr blocks_length,
                bool positive);

  TProxGroupL1(T strength, SArrayULongPtr blocks_start, SArrayULongPtr blocks_length,
                ulong start, ulong end, bool positive);

  inline void set_positive(bool positive);

  inline virtual void set_blocks_start(SArrayULongPtr blocks_start);

  inline virtual void set_blocks_length(SArrayULongPtr blocks_length);

  bool compare(const TProxGroupL1<T> &that);
};

%template(ProxGroupL1Double) TProxGroupL1<double>;
typedef TProxGroupL1<double> ProxGroupL1Double;

%template(ProxGroupL1Float) TProxGroupL1<float>;
typedef TProxGroupL1<float> ProxGroupL1Float;

