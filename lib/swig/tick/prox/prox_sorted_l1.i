// License: BSD 3 clause

%{
#include "tick/prox/prox_slope.h"
%}

template <class T, class K>
class TProxSortedL1 : public TProx<T, K> {
 public:
   TProxSortedL1();
   //TProxSortedL1(T lambda, T fdr, bool positive);

   //TProxSortedL1(T lambda, T fdr, unsigned long start, unsigned long end, bool positive);

   inline T get_weight_i(unsigned long i);

  bool compare(const TProxSortedL1<T, K> &that);
};

%template(ProxSortedL1Double) TProxSortedL1<double, double>;
typedef TProxSortedL1<double, double> ProxSortedL1Double;
TICK_MAKE_TK_PICKLABLE(TProxSortedL1, ProxSortedL1Double, double, double);

%template(ProxSortedL1Float) TProxSortedL1<float, float>;
typedef TProxSortedL1<float, float> ProxSortedL1Float;
TICK_MAKE_TK_PICKLABLE(TProxSortedL1, ProxSortedL1Float, float, float);
