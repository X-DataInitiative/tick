#ifndef LIB_INCLUDE_TICK_PROX_PROX_SEPARABLE_H_
#define LIB_INCLUDE_TICK_PROX_PROX_SEPARABLE_H_

// License: BSD 3 clause

#include "prox.h"

template <class T, class K = T>
class DLL_PUBLIC TProxSeparable : public TProx<T, K> {
 protected:
  using TProx<T, K>::has_range;
  using TProx<T, K>::strength;
  using TProx<T, K>::start;
  using TProx<T, K>::end;
  using TProx<T, K>::positive;

 public:
  using TProx<T, K>::call;

 public:
  TProxSeparable(K strength, bool positive);

  TProxSeparable(K strength, ulong start, ulong end, bool positive);

  std::string get_class_name() const override;

  bool is_separable() const override;


  //! @brief call prox on coeffs, with a given step and store result in out
  //! @note this calls call_single on each coordinate
  void call(const Array<T> &coeffs, K step, Array<T> &out, ulong start,
            ulong end) override;

  //! @brief call prox on coeffs, with a vector of different steps and store result in out
  virtual void call(const Array<T> &coeffs, const Array<K> &step, Array<T> &out);

  //! @brief call prox on a part of coeffs (defined by start-end), with a vector of different
  //! steps and store result in out
  virtual void call(const Array<T> &coeffs, const Array<K> &step, Array<T> &out,
                    ulong start, ulong end);

  //! @brief apply prox on a single value defined by coordinate i
  virtual void call_single(ulong i, const Array<T> &coeffs, K step,
                           Array<T> &out) const;

  //! @brief apply prox on a single value defined by coordinate i several times
  virtual void call_single(ulong i, const Array<T> &coeffs, K step,
                           Array<T> &out, ulong n_times) const;

  K value(const Array<T> &coeffs, ulong start, ulong end) override;

 private:
  //! @brief apply prox on a single value
  virtual K call_single(K x, K step) const;

  //! @brief apply prox on a single value several times
  virtual K call_single(K x, K step, ulong n_times) const;

  //! @brief get penalization value of the prox on a single value
  //! @warning This does not take strength into account
  virtual K value_single(K x) const;

  virtual void set_out_i(Array<T> &out,  size_t i, K k) const;
};

using ProxSeparableDouble = TProxSeparable<double, double>;
using ProxSeparableFloat  = TProxSeparable<float , float>;

#endif  // LIB_INCLUDE_TICK_PROX_PROX_SEPARABLE_H_
