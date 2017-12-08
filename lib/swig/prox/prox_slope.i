// License: BSD 3 clause

%{
#include "tick/prox/prox_slope.h"
%}

class ProxSlope : public Prox {
 public:
   ProxSlope(double lambda, double fdr, bool positive);

   ProxSlope(double lambda, double fdr, unsigned long start, unsigned long end, bool positive);

   inline double get_false_discovery_rate() const;

   inline void set_false_discovery_rate(double fdr);

   inline double get_weight_i(unsigned long i);
};
