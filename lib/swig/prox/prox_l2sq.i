// License: BSD 3 clause

%{
#include "tick/prox/prox_l2sq.h"
%}

template <class T, class K>
class TProxL2Sq : public TProxSeparable<T, K> {
 public:
   TProxL2Sq();
   TProxL2Sq(T strength,
            bool positive);

   TProxL2Sq(T strength,
            ulong start,
            ulong end,
            bool positive);

  bool compare(const TProxL2Sq<T, K> &that);
};

%template(ProxL2SqDouble) TProxL2Sq<double, double>;
typedef TProxL2Sq<double, double> ProxL2SqDouble;
TICK_MAKE_TEMPLATED_PICKLABLE(TProxL2Sq, ProxL2SqDouble , double);

%template(ProxL2SqFloat) TProxL2Sq<float, float>;
typedef TProxL2Sq<float, float> ProxL2SqFloat;
TICK_MAKE_TEMPLATED_PICKLABLE(TProxL2Sq, ProxL2SqFloat , float);
