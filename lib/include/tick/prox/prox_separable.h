#ifndef LIB_INCLUDE_TICK_PROX_PROX_SEPARABLE_H_
#define LIB_INCLUDE_TICK_PROX_PROX_SEPARABLE_H_

// License: BSD 3 clause

#include "prox.h"

template <class T, class K = T>
class DLL_PUBLIC TProxSeparable : public TProx<T, K> {
  // Grants cereal access to default constructor/serialize functions
  friend class cereal::access;

 protected:
  using TProx<T, K>::has_range;
  using TProx<T, K>::strength;
  using TProx<T, K>::start;
  using TProx<T, K>::end;
  using TProx<T, K>::positive;

 public:
  using TProx<T, K>::value;
  using TProx<T, K>::call;
  using TProx<T, K>::get_class_name;
  using TProx<T, K>::is_in_range;

 public:
  // This exists soley for cereal/swig
  TProxSeparable() : TProxSeparable<T, K>(0, 0, 1, 0) {}

  TProxSeparable(T strength, bool positive) : TProx<T, K>(strength, positive) {}

  TProxSeparable(T strength, ulong start, ulong end, bool positive)
      : TProx<T, K>(strength, start, end, positive) {}

  bool is_separable() const override;

  //! @brief call prox on coeffs, with a given step and store result in out
  //! @note this calls call_single on each coordinate
  void call(const Array<K> &coeffs, T step, Array<K> &out, ulong start,
            ulong end) override;

  //! @brief call prox on coeffs, with a vector of different steps and store
  //! result in out
  virtual void call(const Array<K> &coeffs, const Array<T> &step,
                    Array<K> &out);

  //! @brief call prox on a part of coeffs (defined by start-end), with a vector
  //! of different steps and store result in out
  virtual void call(const Array<K> &coeffs, const Array<T> &step, Array<K> &out,
                    ulong start, ulong end);

  //! @brief apply prox on a single value defined by coordinate i
  virtual void call_single(ulong i, const Array<K> &coeffs, T step,
                           Array<K> &out) const;

  //! @brief apply prox on a single value defined by coordinate i several times
  virtual void call_single(ulong i, const Array<K> &coeffs, T step,
                           Array<K> &out, ulong n_times) const;

  T value(const Array<K> &coeffs, ulong start, ulong end) override;

  //! @brief apply prox on a single value
  virtual T call_single(T x, T step) const;

  //! @brief apply prox on a single value with specifying index
  //! @note this is useful for prox that don't apply the exact same operation to
  //! all indexes
  virtual T call_single_with_index(T x, T step, ulong i) const;

 private:
  //! @brief apply prox on a single value several times
  virtual T call_single(T x, T step, ulong n_times) const;

  //! @brief get penalization value of the prox on a single value
  //! @warning This does not take strength into account
  virtual T value_single(T x) const;

 protected:
  template <class Archive>
  void serialize(Archive &ar) {
    ar(cereal::make_nvp("Prox", cereal::base_class<TProx<T, K> >(this)));
  }

  BoolStrReport compare(const TProxSeparable<T, K> &that,
                        std::stringstream &ss) {
    return TProx<T, K>::compare(that, ss);
  }
};

using ProxSeparableDouble = TProxSeparable<double, double>;
CEREAL_SPECIALIZE_FOR_ALL_ARCHIVES(ProxSeparableDouble,
                                   cereal::specialization::member_serialize)

using ProxSeparableFloat = TProxSeparable<float, float>;
CEREAL_SPECIALIZE_FOR_ALL_ARCHIVES(ProxSeparableFloat,
                                   cereal::specialization::member_serialize)

#endif  // LIB_INCLUDE_TICK_PROX_PROX_SEPARABLE_H_
