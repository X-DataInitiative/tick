//
// Created by Martin Bompaire on 26/10/15.
//

#ifndef TICK_OPTIM_PROX_SRC_PROX_H_
#define TICK_OPTIM_PROX_SRC_PROX_H_
#include <memory>
#include "base.h"
#include <string>

class Prox {
 protected:
    // Weight of the proximal operator
    double strength;

    // Flag to know if proximal operator concerns only a part of the vector
    bool has_range;

    // If range is restricted it will be applied from index start to index end
    ulong start, end;

 public:
    explicit Prox(double strength);

    Prox(double strength, ulong start, ulong end);

    virtual const std::string get_class_name() const;

    virtual double value(ArrayDouble &coeffs);

    virtual double _value(ArrayDouble &coeffs,
                          ulong start,
                          ulong end);

    virtual void call(ArrayDouble &coeffs,
                      double step,
                      ArrayDouble &out);

    virtual void call(ArrayDouble &coeffs,
                      ArrayDouble &step,
                      ArrayDouble &out);

    virtual void _call(ArrayDouble &coeffs,
                       double step,
                       ArrayDouble &out,
                       ulong start,
                       ulong end);

  // Compute the prox on the i-th coordinate only
  virtual void _call_i(ulong i,
                       ArrayDouble &coeffs,
                       double step,
                       ArrayDouble &out) const;

  virtual void _call_i(ulong i,
                       ArrayDouble &coeffs,
                       double step,
                       ArrayDouble &out,
                       ulong repeat) const;

  virtual void set_strength(double strength);

    virtual double get_strength() const;

    virtual void set_start_end(ulong start, ulong end);

    ulong get_start();
    ulong get_end();
};

typedef std::shared_ptr<Prox> ProxPtr;

#endif  // TICK_OPTIM_PROX_SRC_PROX_H_
