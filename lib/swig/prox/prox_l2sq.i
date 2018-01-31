// License: BSD 3 clause

%{
#include "tick/prox/prox_l2sq.h"
%}

template <class T, class K>
class TProxL2Sq : public TProxSeparable<T, K> {
 public:
   TProxL2Sq(K strength,
            bool positive);

   TProxL2Sq(K strength,
            ulong start,
            ulong end,
            bool positive);
};

%rename(ProxL2SqDouble) TProxL2Sq<double, double>;
class TProxL2Sq<double, double> : public TProxSeparable<double, double> {
 public:
   TProxL2Sq(double strength,
            bool positive);

   TProxL2Sq(double strength,
            ulong start,
            ulong end,
            bool positive);
};
typedef TProxL2Sq<double, double> ProxL2SqDouble;

%rename(ProxL2SqFloat) TProxL2Sq<float, float>;
class TProxL2Sq<float, float> : public TProxSeparable<float, float> {
 public:
   TProxL2Sq(float strength,
            bool positive);

   TProxL2Sq(float strength,
            ulong start,
            ulong end,
            bool positive);
};
typedef TProxL2Sq<float, float> ProxL2SqFloat;
