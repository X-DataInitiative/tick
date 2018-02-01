//
// Created by Martin Bompaire on 26/10/15.
//

#ifndef LIB_INCLUDE_TICK_PROX_PROX_H_
#define LIB_INCLUDE_TICK_PROX_PROX_H_

// License: BSD 3 clause

#include "tick/base/base.h"

#include <memory>
#include <string>

template <class T, class K = T>
class DLL_PUBLIC TProx {
 protected:
  //! @brief Flag to know if proximal operator concerns only a part of the vector
  bool has_range;

  //! @brief If true, we apply on non negativity constraint
  bool positive;

  //! @brief If range is restricted it will be applied from index start to index end
  ulong start, end;

  //! @brief Weight of the proximal operator
  K strength;

 public:
  TProx(K strength, bool positive);

  TProx(K strength, ulong start, ulong end, bool positive);

  virtual ~TProx() {}

  virtual std::string get_class_name() const;

  virtual bool is_separable() const;

  //! @brief call prox on coeffs, with a given step and store result in out
  virtual void call(const Array<T> &coeffs, K step, Array<T> &out);

  //! @brief call prox on a part of coeffs (defined by start-end), with a given step and
  //! store result in out
  virtual void call(
    const Array<T> &coeffs,
    K step,
    Array<T> &out,
    ulong start,
    ulong end);

  //! @brief get penalization value of the prox on the coeffs vector.
  //! This takes strength into account
  virtual K value(const Array<T> &coeffs);

  //! @brief get penalization value of the prox on a part of coeffs (defined by start-end).
  //! This takes strength into account
  virtual K value(
    const Array<T> &coeffs,
    ulong start,
    ulong end);

  virtual K get_strength() const;

  virtual void set_strength(K strength);

  virtual ulong get_start() const;

  virtual ulong get_end() const;

  virtual void set_start_end(ulong start, ulong end);

  virtual bool get_positive() const;

  virtual void set_positive(bool positive);
};

using ProxDouble = TProx<double, double>;
using ProxDoublePtr = std::shared_ptr<ProxDouble>;
using ProxDoublePtrVector = std::vector<ProxDoublePtr>;

using ProxFloat  = TProx<float , float>;
using ProxFloatPtr = std::shared_ptr<ProxFloat>;

class Prox : public TProx<double, double> {
 public:
  Prox(double strength, bool positive)
    : TProx<double, double>(strength, positive) {}

  Prox(double strength, ulong start, ulong end, bool positive)
    : TProx<double, double>(strength, start, end, positive) {}
};

#endif  // LIB_INCLUDE_TICK_PROX_PROX_H_
