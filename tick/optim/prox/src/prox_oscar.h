#ifndef TICK_OPTIM_PROX_SRC_PROX_OSCAR_H_
#define TICK_OPTIM_PROX_SRC_PROX_OSCAR_H_

// License: BSD 3 clause

#include "prox_sorted_l1.h"

class ProxOscar : public ProxSortedL1 {
 protected:
  void compute_weights(void) override;
  double ratio;

 public:
  ProxOscar(double strength, double ratio, bool positive);

  ProxOscar(double strength,
            double ratio,
            ulong start,
            ulong end,
            bool positive);

  const std::string get_class_name() const override;

  virtual double get_ratio() const;

  virtual void set_ratio(double ratio);
};

#endif  // TICK_OPTIM_PROX_SRC_PROX_OSCAR_H_
