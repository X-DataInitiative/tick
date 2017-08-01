// License: BSD 3 clause

%{
#include "prox_oscar.h"
%}

class ProxOscar : public Prox {
 public:
   ProxOscar(double strength, double ratio, bool positive);

   ProxOscar(double strength, double ratio, unsigned long start, unsigned long end, bool positive);

   virtual double get_ratio() const final;

   virtual void set_ratio(double ratio) final;
};
