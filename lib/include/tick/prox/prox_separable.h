#ifndef LIB_INCLUDE_TICK_PROX_PROX_SEPARABLE_H_
#define LIB_INCLUDE_TICK_PROX_PROX_SEPARABLE_H_

// License: BSD 3 clause

#include "prox.h"

template <class T>
class DLL_PUBLIC TProxSeparable : public TProx<T> {
  // Grants cereal access to default constructor
  friend class cereal::access;

 protected:
  using TProx<T>::has_range;
  using TProx<T>::strength;
  using TProx<T>::start;
  using TProx<T>::end;
  using TProx<T>::positive;

 public:
  using TProx<T>::call;
  using TProx<T>::get_class_name;

 protected:
  // This exists soley for cereal/swig
  TProxSeparable() : TProxSeparable<T>(0, 0, 1, 0) {}

 public:
  TProxSeparable(T strength, bool positive) : TProx<T>(strength, positive) {}

  TProxSeparable(T strength, ulong start, ulong end, bool positive)
      : TProx<T>(strength, start, end, positive) {}

  bool is_separable() const override;

  //! @brief call prox on coeffs, with a given step and store result in out
  //! @note this calls call_single on each coordinate
  void call(const Array<T> &coeffs, T step, Array<T> &out, ulong start,
            ulong end) override;

  //! @brief call prox on coeffs, with a vector of different steps and store
  //! result in out
  virtual void call(const Array<T> &coeffs, const Array<T> &step,
                    Array<T> &out);

  //! @brief call prox on a part of coeffs (defined by start-end), with a vector
  //! of different steps and store result in out
  virtual void call(const Array<T> &coeffs, const Array<T> &step, Array<T> &out,
                    ulong start, ulong end);

  //! @brief apply prox on a single value defined by coordinate i
  virtual void call_single(ulong i, const Array<T> &coeffs, T step,
                           Array<T> &out) const;

  //! @brief apply prox on a single value defined by coordinate i several times
  virtual void call_single(ulong i, const Array<T> &coeffs, T step,
                           Array<T> &out, ulong n_times) const;

  T value(const Array<T> &coeffs, ulong start, ulong end) override;

 private:
  //! @brief apply prox on a single value
  virtual T call_single(T x, T step) const;

  //! @brief apply prox on a single value several times
  virtual T call_single(T x, T step, ulong n_times) const;

  //! @brief get penalization value of the prox on a single value
  //! @warning This does not take strength into account
  virtual T value_single(T x) const;

 protected:
  template <class Archive>
  void serialize(Archive &ar) {
    ar(cereal::make_nvp("Prox", cereal::base_class<TProx<T> >(this)));
  }

  BoolStrReport compare(const TProxSeparable<T> &that, std::stringstream &ss) {
    return TProx<T>::compare(that, ss);
  }
};

using ProxSeparable = TProxSeparable<double>;

using ProxSeparableDouble = TProxSeparable<double>;

using ProxSeparableFloat = TProxSeparable<float>;

#endif  // LIB_INCLUDE_TICK_PROX_PROX_SEPARABLE_H_
