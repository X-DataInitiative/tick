// License: BSD 3 clause

%{
#include "tick/prox/prox_binarsity.h"
%}

template <class T>
class TProxWithGroups : public TProx<T> {
 public:
  TProxWithGroups(T strength, SArrayULongPtr blocks_start, SArrayULongPtr blocks_length,
                 bool positive);

  TProxWithGroups(T strength, SArrayULongPtr blocks_start, SArrayULongPtr blocks_length,
                 ulong start, ulong end, bool positive);
};

%template(ProxWithGroups) TProxWithGroups<double>;
typedef TProxWithGroups<double> ProxWithGroups;

%template(ProxWithGroupsDouble) TProxWithGroups<double>;
typedef TProxWithGroups<double> ProxWithGroupsDouble;

%template(ProxWithGroupsFloat) TProxWithGroups<float>;
typedef TProxWithGroups<float> ProxWithGroupsFloat;
