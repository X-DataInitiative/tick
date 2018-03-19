// License: BSD 3 clause

%{
#include "tick/prox/prox_l1.h"
%}

template <class T>
class TProxL1 : public TProx<T> {
 public:
   TProxL1();
   TProxL1(double strength,
          bool positive);

   TProxL1(double strength,
          ulong start,
          ulong end,
          bool positive);
          
  bool compare(const TProxL1<T> &that);
};

%rename(ProxL1Double) TProxL1<double>;
class ProxL1Double : public TProxSeparable<double> {
 public:
   ProxL1Double();
   ProxL1Double(double strength,
            bool positive);

   ProxL1Double(double strength,
            ulong start,
            ulong end,
            bool positive);

  bool compare(const ProxL1Double &that);
};
typedef TProxL1<double> ProxL1Double;
TICK_MAKE_PICKLABLE(ProxL1Double);

%rename(ProxL1Float) TProxL1<float>;
class ProxL1Float : public TProxSeparable<float> {
 public:
   ProxL1Float();
   ProxL1Float(float strength,
            bool positive);

   ProxL1Float(float strength,
            ulong start,
            ulong end,
            bool positive);
          
  bool compare(const ProxL1Float &that);
};
typedef TProxL1<float> ProxL1Float;
TICK_MAKE_PICKLABLE(ProxL1Float);
