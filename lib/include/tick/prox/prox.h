//
// Created by Martin Bompaire on 26/10/15.
//

#ifndef LIB_INCLUDE_TICK_PROX_PROX_H_
#define LIB_INCLUDE_TICK_PROX_PROX_H_

// License: BSD 3 clause

#include "tick/array/array.h"
#include "tick/base/base.h"
#include "tick/base/serialization.h"

#include <memory>
#include <string>

template <class T, class K = T>
class DLL_PUBLIC TProx {
  // Grants cereal access to default constructor/serialize functions
  friend class cereal::access;

  template <class T1, class K1>
  friend std::ostream& operator<<(std::ostream&, const TProx<T1, K1>&);

 protected:
  //! @brief Flag to know if proximal operator concerns only a part of the
  //! vector
  bool has_range = false;

  //! @brief If true, we apply on non negativity constraint
  bool positive = false;

  //! @brief If range is restricted it will be applied from index start to index
  //! end
  ulong start = 0, end = 0;

  //! @brief Weight of the proximal operator
  T strength;

 public:
  // This exists soley for cereal/swig
  TProx() {}

  TProx(T strength, bool positive);
  TProx(T strength, ulong start, ulong end, bool positive);

  virtual ~TProx() {}

  virtual const std::string get_class_name() const {
    std::stringstream ss;
    ss << typeid(*this).name() << "<" << typeid(T).name() << ">";
    return ss.str();
  }

  virtual bool is_separable() const;

  //! @brief call prox on coeffs, with a given step and store result in out
  virtual void call(const Array<K>& coeffs, T step, Array<K>& out);

  //! @brief call prox on a part of coeffs (defined by start-end), with a given
  //! step and store result in out
  virtual void call(const Array<K>& coeffs, T step, Array<K>& out, ulong start,
                    ulong end);

  //! @brief get penalization value of the prox on the coeffs vector.
  //! This takes strength into account
  virtual T value(const Array<K>& coeffs);

  //! @brief get penalization value of the prox on a part of coeffs (defined by
  //! start-end). This takes strength into account
  virtual T value(const Array<K>& coeffs, ulong start, ulong end);

  virtual T get_strength() const;

  virtual void set_strength(T strength);

  virtual ulong get_start() const;

  virtual ulong get_end() const;

  virtual void set_start_end(ulong start, ulong end);

  virtual bool get_positive() const;

  virtual void set_positive(bool positive);

  template <class Archive>
  void serialize(Archive& ar) {
    ar(CEREAL_NVP(strength), CEREAL_NVP(has_range), CEREAL_NVP(start),
       CEREAL_NVP(end), CEREAL_NVP(positive));
  }

  BoolStrReport operator==(const TProx<T, K>& that) = delete;

  BoolStrReport compare(const TProx<T, K>& that, std::stringstream& ss) {
    return BoolStrReport(
        TICK_CMP_REPORT(ss, strength) && TICK_CMP_REPORT(ss, has_range) &&
            TICK_CMP_REPORT(ss, start) && TICK_CMP_REPORT(ss, end) &&
            TICK_CMP_REPORT(ss, positive),
        ss.str());
  }
};

template <typename T, typename K>
inline std::ostream& operator<<(std::ostream& s, const TProx<T, K>& p) {
  return s << typeid(p).name() << "<" << typeid(T).name() << ">";
}

using ProxDouble = TProx<double, double>;
using ProxDoublePtr = std::shared_ptr<ProxDouble>;
using ProxDoublePtrVector = std::vector<ProxDoublePtr>;
CEREAL_SPECIALIZE_FOR_ALL_ARCHIVES(ProxDouble,
                                   cereal::specialization::member_serialize)

using ProxFloat = TProx<float, float>;
using ProxFloatPtr = std::shared_ptr<ProxFloat>;
using ProxFloatPtrVector = std::vector<ProxFloatPtr>;

CEREAL_SPECIALIZE_FOR_ALL_ARCHIVES(ProxFloat,
                                   cereal::specialization::member_serialize)

using ProxAtomicDouble = TProx<double, std::atomic<double>>;
using ProxAtomicDoublePtr = std::shared_ptr<ProxDouble>;
using ProxAtomicDoublePtrVector = std::vector<ProxDoublePtr>;

using ProxAtomicFloat = TProx<float, std::atomic<float>>;
using ProxAtomicFloatPtr = std::shared_ptr<ProxFloat>;
using ProxAtomicFloatPtrVector = std::vector<ProxFloatPtr>;

#endif  // LIB_INCLUDE_TICK_PROX_PROX_H_
