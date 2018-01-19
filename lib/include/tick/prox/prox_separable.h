#ifndef LIB_INCLUDE_TICK_PROX_PROX_SEPARABLE_H_
#define LIB_INCLUDE_TICK_PROX_PROX_SEPARABLE_H_

// License: BSD 3 clause

#include "prox.h"

class DLL_PUBLIC ProxSeparable : public Prox {
 public:
  ProxSeparable(double strength, bool positive);

  ProxSeparable(double strength, ulong start, ulong end, bool positive);

  const std::string get_class_name() const override;

  const bool is_separable() const override;

  using Prox::call;

  //! @brief call prox on coeffs, with a given step and store result in out
  //! @note this calls call_single on each coordinate
  void call(const ArrayDouble &coeffs, double step, ArrayDouble &out, ulong start,
            ulong end) override;

  //! @brief call prox on coeffs, with a vector of different steps and store result in out
  virtual void call(const ArrayDouble &coeffs, const ArrayDouble &step, ArrayDouble &out);

  //! @brief call prox on a part of coeffs (defined by start-end), with a vector of different
  //! steps and store result in out
  virtual void call(const ArrayDouble &coeffs, const ArrayDouble &step, ArrayDouble &out,
                    ulong start, ulong end);

  //! @brief apply prox on a single value defined by coordinate i
  virtual void call_single(ulong i, const ArrayDouble &coeffs, double step,
                           ArrayDouble &out) const;

  //! @brief apply prox on a single value defined by coordinate i several times
  virtual void call_single(ulong i, const ArrayDouble &coeffs, double step,
                           ArrayDouble &out, ulong n_times) const;

  double value(const ArrayDouble &coeffs, ulong start, ulong end) override;

 private:
  //! @brief apply prox on a single value
  virtual double call_single(double x, double step) const;

  //! @brief apply prox on a single value several times
  virtual double call_single(double x, double step, ulong n_times) const;

  //! @brief get penalization value of the prox on a single value
  //! @warning This does not take strength into account
  virtual double value_single(double x) const;
};

#endif  // LIB_INCLUDE_TICK_PROX_PROX_SEPARABLE_H_
