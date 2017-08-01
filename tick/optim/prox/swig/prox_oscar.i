// License: BSD 3 clause

%{
#include "prox_oscar.h"
%}

class ProxOscar : public Prox {
 public:
   ProxOscar(double strength, double ratio, bool positive);

   ProxOscar(double strength, double ratio, unsigned long start, unsigned long end, bool positive);

   virtual double get_ratio() const;

   virtual void set_ratio(double ratio);

   inline double get_weight_i(unsigned long i) const;
};
