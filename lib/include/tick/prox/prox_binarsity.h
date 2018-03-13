#ifndef LIB_INCLUDE_TICK_PROX_PROX_BINARSITY_H_
#define LIB_INCLUDE_TICK_PROX_PROX_BINARSITY_H_

// License: BSD 3 clause

#include "prox_with_groups.h"

template <class T>
class DLL_PUBLIC TProxBinarsity : public TProxWithGroups<T> {
  // Grants cereal access to default constructor
  friend class cereal::access;

 protected:
  using TProxWithGroups<T>::proxs;
  using TProxWithGroups<T>::is_synchronized;
  using TProxWithGroups<T>::synchronize_proxs;

 public:
  using TProxWithGroups<T>::get_class_name;

 protected:
  // This exists soley for cereal which has friend access
  TProxBinarsity() : TProxBinarsity(0, nullptr, nullptr, false) {}

  std::unique_ptr<TProx<T> > build_prox(T strength, ulong start, ulong end,
                                        bool positive) override;

 public:
  TProxBinarsity(T strength, SArrayULongPtr blocks_start,
                 SArrayULongPtr blocks_length, bool positive);

  TProxBinarsity(T strength, SArrayULongPtr blocks_start,
                 SArrayULongPtr blocks_length, ulong start, ulong end,
                 bool positive);

  void call(const Array<T>& coeffs, T step, Array<T>& out, ulong start,
            ulong end) override;

  template <class Archive>
  void serialize(Archive& ar) {
    ar(cereal::make_nvp("ProxWithGroups", cereal::base_class<TProx<T> >(this)));
  }

  BoolStrReport compare(const TProxBinarsity<T>& that) {
    std::stringstream ss;
    ss << get_class_name();
    auto are_equal = TProxWithGroups<T>::compare(that, ss);
    return BoolStrReport(are_equal, ss.str());
  }
  BoolStrReport operator==(const TProxBinarsity<T>& that) {
    return compare(that);
  }
};

using ProxBinarsity = TProxBinarsity<double>;

using ProxBinarsityDouble = TProxBinarsity<double>;
CEREAL_SPECIALIZE_FOR_ALL_ARCHIVES(ProxBinarsityDouble,
                                   cereal::specialization::member_serialize)
CEREAL_REGISTER_TYPE(ProxBinarsityDouble)

using ProxBinarsityFloat = TProxBinarsity<float>;
CEREAL_SPECIALIZE_FOR_ALL_ARCHIVES(ProxBinarsityFloat,
                                   cereal::specialization::member_serialize)
CEREAL_REGISTER_TYPE(ProxBinarsityFloat)

#endif  // LIB_INCLUDE_TICK_PROX_PROX_BINARSITY_H_
