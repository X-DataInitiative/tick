// License: BSD 3 clause

%{
#include "tick/prox/prox_l1.h"
%}

template <class T, class K>
class TProxL1 : public TProx<T, K> {
 public:
   ProxL1(double strength,
          bool positive);

   ProxL1(double strength,
          ulong start,
          ulong end,
          bool positive);
};


%rename(ProxL1Double) TProxL1<double, double>;
class TProxL1<double, double> : public TProxSeparable<double, double> {
 public:
   TProxL1(double strength,
            bool positive);

   TProxL1(double strength,
            ulong start,
            ulong end,
            bool positive);
};
typedef TProxL1<double, double> ProxL1Double;

%rename(ProxL1Float) TProxL1<float, float>;
class TProxL1<float, float> : public TProxSeparable<float, float> {
 public:
   TProxL1(float strength,
            bool positive);

   TProxL1(float strength,
            ulong start,
            ulong end,
            bool positive);
};
typedef TProxL1<float, float> ProxL1Float;
