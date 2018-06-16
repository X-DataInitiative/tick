// License: BSD 3 clause

%{
#include "tick/prox/prox_slope.h"
%}

template <class T, class K>
class TProxSlope : public TProx<T, K> {
 public:
   TProxSlope(T lambda, T fdr, bool positive);

   TProxSlope(T lambda, T fdr, unsigned long start, unsigned long end, bool positive);

   inline T get_false_discovery_rate() const;

   inline void set_false_discovery_rate(T fdr);

   inline T get_weight_i(unsigned long i);

  bool compare(const TProxSlope<T, K> &that);
};

%template(ProxSlope) TProxSlope<double, double>;
typedef TProxSlope<double, double> ProxSlope;

%template(ProxSlopeDouble) TProxSlope<double, double>;
typedef TProxSlope<double, double> ProxSlopeDouble;

%template(ProxSlopeFloat) TProxSlope<float, float>;
typedef TProxSlope<float, float> ProxSlopeFloat;
