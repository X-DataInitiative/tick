//
// Created by Martin Bompaire on 26/10/15.
//

#ifndef LIB_INCLUDE_TICK_PROX_PROX_H_
#define LIB_INCLUDE_TICK_PROX_PROX_H_

// License: BSD 3 clause

#include "tick/base/base.h"
#include "tick/base/serialization.h"

#include <memory>
#include <string>

template <class T>
class DLL_PUBLIC TProx {
  // Grants cereal access to default constructor
  friend class cereal::access;

  template <class T1>
  friend std::ostream& operator<<(std::ostream&, const TProx<T1>&);

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

 protected:
  // This exists soley for cereal/swig
  TProx() {}

 public:
  TProx(T strength, bool positive);
  TProx(T strength, ulong start, ulong end, bool positive);

  TProx(const TProx<T>& other) = delete;
  TProx(TProx<T>&& other) = delete;
  TProx<T>& operator=(const TProx<T>& other) = delete;
  TProx<T>& operator=(TProx<T>&& other) = delete;

  virtual ~TProx() {}

  virtual const std::string get_class_name() const {
    std::stringstream ss;
    ss << typeid(*this).name() << "<" << typeid(T).name() << ">";
    return ss.str();
  }

  virtual bool is_separable() const;

  //! @brief call prox on coeffs, with a given step and store result in out
  virtual void call(const Array<T>& coeffs, T step, Array<T>& out);

  //! @brief call prox on a part of coeffs (defined by start-end), with a given
  //! step and store result in out
  virtual void call(const Array<T>& coeffs, T step, Array<T>& out, ulong start,
                    ulong end);

  //! @brief get penalization value of the prox on the coeffs vector.
  //! This takes strength into account
  virtual T value(const Array<T>& coeffs);

  //! @brief get penalization value of the prox on a part of coeffs (defined by
  //! start-end). This takes strength into account
  virtual T value(const Array<T>& coeffs, ulong start, ulong end);

  virtual T get_strength() const;

  virtual void set_strength(T strength);

  virtual ulong get_start() const;

  virtual ulong get_end() const;

  virtual void set_start_end(ulong start, ulong end);

  virtual bool get_positive() const;

  virtual void set_positive(bool positive);

 protected:
  template <class Archive>
  void serialize(Archive& ar) {
    ar(CEREAL_NVP(strength), CEREAL_NVP(has_range), CEREAL_NVP(start),
       CEREAL_NVP(end), CEREAL_NVP(positive));
  }

  BoolStrReport operator==(const TProx<T>& that) = delete;

  BoolStrReport compare(const TProx<T> &that, std::stringstream& ss) {
    return BoolStrReport(
        TICK_CMP_REPORT(ss, strength) && TICK_CMP_REPORT(ss, has_range) &&
            TICK_CMP_REPORT(ss, start) && TICK_CMP_REPORT(ss, end) &&
            TICK_CMP_REPORT(ss, positive),
        ss.str());
  }
};

template <typename T>
inline std::ostream& operator<<(std::ostream& s, const TProx<T>& p) {
  return s << typeid(p).name() << "<" << typeid(T).name() << ">";
}

using ProxDouble = TProx<double>;
using ProxDoublePtr = std::shared_ptr<ProxDouble>;
using ProxDoublePtrVector = std::vector<ProxDoublePtr>;

using ProxFloat = TProx<float>;
using ProxFloatPtr = std::shared_ptr<ProxFloat>;
using ProxFloatPtrVector = std::vector<ProxFloatPtr>;

#endif  // LIB_INCLUDE_TICK_PROX_PROX_H_
