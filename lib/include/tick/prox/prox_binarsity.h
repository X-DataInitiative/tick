#ifndef LIB_INCLUDE_TICK_PROX_PROX_BINARSITY_H_
#define LIB_INCLUDE_TICK_PROX_PROX_BINARSITY_H_

// License: BSD 3 clause

#include "prox_with_groups.h"

template <class T>
class DLL_PUBLIC TProxBinarsity : public TProxWithGroups<T> {
 protected:
  using TProxWithGroups<T>::proxs;
  using TProxWithGroups<T>::is_synchronized;
  using TProxWithGroups<T>::synchronize_proxs;

 public:
  using TProxWithGroups<T>::get_class_name;

 protected:
  std::unique_ptr<TProx<T> > build_prox(T strength, ulong start, ulong end,
                                        bool positive) override;

 public:
  // This exists soley for cereal/swig
  TProxBinarsity() : TProxBinarsity(0, nullptr, nullptr, false) {}

  TProxBinarsity(T strength, SArrayULongPtr blocks_start,
                 SArrayULongPtr blocks_length, bool positive)
      : TProxWithGroups<T>(strength, blocks_start, blocks_length, positive) {}

  TProxBinarsity(T strength, SArrayULongPtr blocks_start,
                 SArrayULongPtr blocks_length, ulong start, ulong end,
                 bool positive)
      : TProxWithGroups<T>(strength, blocks_start, blocks_length, start, end,
                           positive) {}

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
using ProxBinarsityFloat = TProxBinarsity<float>;

CEREAL_SPECIALIZE_FOR_ALL_ARCHIVES(ProxBinarsityDouble,
                                   cereal::specialization::member_serialize)
CEREAL_REGISTER_TYPE(ProxBinarsityDouble)


CEREAL_SPECIALIZE_FOR_ALL_ARCHIVES(ProxBinarsityFloat,
                                   cereal::specialization::member_serialize)
CEREAL_REGISTER_TYPE(ProxBinarsityFloat)

#endif  // LIB_INCLUDE_TICK_PROX_PROX_BINARSITY_H_
