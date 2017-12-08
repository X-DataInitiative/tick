// License: BSD 3 clause

%{
#include "tick/prox/prox_separable.h"
%}

class ProxSeparable : public Prox {

 public:
  ProxSeparable(double strength,
                bool positive);

  ProxSeparable(double strength,
                unsigned long start,
                unsigned long end,
                bool positive);

  using Prox::call;

  virtual void call(const ArrayDouble &coeffs,
                    const ArrayDouble &step,
                    ArrayDouble &out);
};
