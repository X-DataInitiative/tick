// License: BSD 3 clause

%{
#include "tick/prox/prox_l1.h"
%}

template <class T>
class TProxL1 : public TProx<T> {
 public:
   ProxL1(double strength,
          bool positive);

   ProxL1(double strength,
          ulong start,
          ulong end,
          bool positive);
};

%rename(ProxL1) TProxL1<double>;
class TProxL1<double> : public TProxSeparable<double> {
 public:
   TProxL1(double strength,
            bool positive);

   TProxL1(double strength,
            ulong start,
            ulong end,
            bool positive);
};
typedef TProxL1<double> ProxL1;

%rename(ProxL1Double) TProxL1<double>;
class TProxL1<double> : public TProxSeparable<double> {
 public:
   TProxL1(double strength,
            bool positive);

   TProxL1(double strength,
            ulong start,
            ulong end,
            bool positive);
};
typedef TProxL1<double> ProxL1Double;

%rename(ProxL1Float) TProxL1<float>;
class TProxL1<float> : public TProxSeparable<float> {
 public:
   TProxL1(float strength,
            bool positive);

   TProxL1(float strength,
            ulong start,
            ulong end,
            bool positive);
};
typedef TProxL1<float> ProxL1Float;
