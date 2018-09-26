// License: BSD 3 clause

%{
#include "tick/prox/prox_zero.h"
%}

template <class T, class K>
class TProxZero : public TProx<T, K> {
 public:
   TProxZero();
   TProxZero(T strength);

   TProxZero(T strength,
            unsigned long start,
            unsigned long end);

  bool compare(const TProxZero<T, K> &that);
};

%template(ProxZeroDouble) TProxZero<double, double>;
typedef TProxZero<double, double> ProxZeroDouble;
TICK_MAKE_TEMPLATED_PICKLABLE(TProxZero, ProxZeroDouble, double);

%template(ProxZeroFloat) TProxZero<float, float>;
typedef TProxZero<float, float> ProxZeroFloat;
TICK_MAKE_TEMPLATED_PICKLABLE(TProxZero, ProxZeroFloat , float);


%template(ProxZeroAtomicDouble) TProxZero<double, std::atomic<double>>;
typedef TProxZero<double, std::atomic<double>> ProxZeroAtomicDouble;

%template(ProxZeroAtomicFloat) TProxZero<float, std::atomic<float>>;
typedef TProxZero<float, std::atomic<float>> ProxZeroAtomicFloat;

