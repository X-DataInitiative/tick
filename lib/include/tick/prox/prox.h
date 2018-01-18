//
// Created by Martin Bompaire on 26/10/15.
//

#ifndef LIB_INCLUDE_TICK_PROX_PROX_H_
#define LIB_INCLUDE_TICK_PROX_PROX_H_

// License: BSD 3 clause

#include "tick/base/base.h"

#include <memory>
#include <string>

class DLL_PUBLIC Prox {
 protected:
  //! @brief Weight of the proximal operator
  double strength;

  //! @brief Flag to know if proximal operator concerns only a part of the vector
  bool has_range;

  //! @brief If range is restricted it will be applied from index start to index end
  ulong start, end;

  //! @brief If true, we apply on non negativity constraint
  bool positive;

 public:
  Prox(double strength, bool positive);

  Prox(double strength, ulong start, ulong end, bool positive);

  virtual const std::string get_class_name() const;

  virtual const bool is_separable() const;

  //! @brief call prox on coeffs, with a given step and store result in out
  virtual void call(const ArrayDouble &coeffs, double step, ArrayDouble &out);

  //! @brief call prox on a part of coeffs (defined by start-end), with a given step and
  //! store result in out
  virtual void call(const ArrayDouble &coeffs,
                    double step,
                    ArrayDouble &out,
                    ulong start,
                    ulong end);

  //! @brief get penalization value of the prox on the coeffs vector.
  //! This takes strength into account
  virtual double value(const ArrayDouble &coeffs);

  //! @brief get penalization value of the prox on a part of coeffs (defined by start-end).
  //! This takes strength into account
  virtual double value(const ArrayDouble &coeffs,
                       ulong start,
                       ulong end);

  virtual double get_strength() const;

  virtual void set_strength(double strength);

  virtual ulong get_start() const;

  virtual ulong get_end() const;

  virtual void set_start_end(ulong start,
                             ulong end);

  virtual bool get_positive() const;

  virtual void set_positive(bool positive);
};

typedef std::shared_ptr<Prox> ProxPtr;

#endif  // LIB_INCLUDE_TICK_PROX_PROX_H_
