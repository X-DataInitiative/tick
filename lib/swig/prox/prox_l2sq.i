// License: BSD 3 clause

%{
#include "tick/prox/prox_l2sq.h"
%}

template <class T>
class TProxL2Sq : public TProxSeparable<T> {
 public:
   TProxL2Sq(T strength,
            bool positive);

   TProxL2Sq(T strength,
            ulong start,
            ulong end,
            bool positive);
};

%template(ProxL2Sq) TProxL2Sq<double>;
typedef TProxL2Sq<double> ProxL2Sq;

%template(ProxL2SqDouble) TProxL2Sq<double>;
typedef TProxL2Sq<double> ProxL2SqDouble;

%template(ProxL2SqFloat) TProxL2Sq<float>;
typedef TProxL2Sq<float> ProxL2SqFloat;
