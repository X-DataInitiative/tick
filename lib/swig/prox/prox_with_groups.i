// License: BSD 3 clause

%{
#include "tick/prox/prox_binarsity.h"
%}

template <class T, class K>
class TProxWithGroups : public TProx<T, K> {
 public:
  TProxWithGroups(T strength, SArrayULongPtr blocks_start, SArrayULongPtr blocks_length,
                 bool positive);

  TProxWithGroups(T strength, SArrayULongPtr blocks_start, SArrayULongPtr blocks_length,
                 ulong start, ulong end, bool positive);
};

%template(ProxWithGroupsDouble) TProxWithGroups<double, double>;
typedef TProxWithGroups<double, double> ProxWithGroupsDouble;

%template(ProxWithGroupsFloat) TProxWithGroups<float, float>;
typedef TProxWithGroups<float, float> ProxWithGroupsFloat;
