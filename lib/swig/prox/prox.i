// License: BSD 3 clause

%{
#include "tick/prox/prox.h"
%}

class Prox {
 public:
  Prox(double strength,
       bool positive);

  Prox(double strength,
       unsigned long start,
       unsigned long end,
       bool positive);

  virtual void call(const ArrayDouble &coeffs,
                    double step,
                    ArrayDouble &out);

  virtual double value(const ArrayDouble &coeffs);

  virtual double get_strength() const;

  virtual void set_strength(double strength);

  virtual ulong get_start() const final;

  virtual ulong get_end() const final;

  virtual void set_start_end(ulong start, ulong end);

  virtual bool get_positive() const final;

  virtual void set_positive(bool positive) final;
};

typedef std::shared_ptr<Prox> ProxPtr;
