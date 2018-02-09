// License: BSD 3 clause

%{
#include "tick/prox/prox_slope.h"
%}

template <class T>
class TProxSlope : public TProx<T> {
 public:
   TProxSlope(T lambda, T fdr, bool positive);

   TProxSlope(T lambda, T fdr, unsigned long start, unsigned long end, bool positive);

   inline T get_false_discovery_rate() const;

   inline void set_false_discovery_rate(T fdr);

   inline T get_weight_i(unsigned long i);
};

%template(ProxSlope) TProxSlope<double>;
typedef TProxSlope<double> ProxSlope;

%template(ProxSlopeDouble) TProxSlope<double>;
typedef TProxSlope<double> ProxSlopeDouble;

%template(ProxSlopeFloat) TProxSlope<float>;
typedef TProxSlope<float> ProxSlopeFloat;
