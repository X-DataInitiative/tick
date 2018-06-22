#ifndef LIB_INCLUDE_TICK_PROX_PROX_BINARSITY_H_
#define LIB_INCLUDE_TICK_PROX_PROX_BINARSITY_H_

// License: BSD 3 clause

#include "prox_with_groups.h"
#include "tick/prox/prox_tv.h"

template <class T, class K = T>
class DLL_PUBLIC TProxBinarsity : public TProxWithGroups<T, K> {
 protected:
  using TProxWithGroups<T, K>::proxs;
  using TProxWithGroups<T, K>::is_synchronized;
  using TProxWithGroups<T, K>::synchronize_proxs;

 public:
  using TProxWithGroups<T, K>::get_class_name;

 protected:
  std::unique_ptr<TProx<T, K> > build_prox(T strength, ulong start, ulong end,
                                           bool positive) override;

 public:
  // This exists soley for cereal/swig
  TProxBinarsity() : TProxBinarsity(0, nullptr, nullptr, false) {}

  TProxBinarsity(T strength, SArrayULongPtr blocks_start,
                 SArrayULongPtr blocks_length, bool positive)
      : TProxWithGroups<T, K>(strength, blocks_start, blocks_length, positive) {
  }

  TProxBinarsity(T strength, SArrayULongPtr blocks_start,
                 SArrayULongPtr blocks_length, ulong start, ulong end,
                 bool positive)
      : TProxWithGroups<T, K>(strength, blocks_start, blocks_length, start, end,
                              positive) {}

  // There's something odd on windows trying to copy the unique_ptr in the
  // superclass
  TProxBinarsity(const TProxBinarsity&) = delete;
  TProxBinarsity(const TProxBinarsity&&) = delete;
  TProxBinarsity& operator=(const TProxBinarsity&) = delete;
  TProxBinarsity& operator=(const TProxBinarsity&&) = delete;

  void call(const Array<K>& coeffs, T step, Array<K>& out, ulong start,
            ulong end) override;

  template <class Archive>
  void serialize(Archive& ar) {
    ar(cereal::make_nvp("ProxWithGroups",
                        cereal::base_class<TProx<T, K> >(this)));
  }

  BoolStrReport compare(const TProxBinarsity<T, K>& that) {
    std::stringstream ss;
    ss << get_class_name();
    auto are_equal = TProxWithGroups<T, K>::compare(that, ss);
    return BoolStrReport(are_equal, ss.str());
  }
  BoolStrReport operator==(const TProxBinarsity<T, K>& that) {
    return compare(that);
  }
};

template <class T, class K>
std::unique_ptr<TProx<T, K> > TProxBinarsity<T, K>::build_prox(T strength,
                                                               ulong start,
                                                               ulong end,
                                                               bool positive) {
  return std::move(std::unique_ptr<TProxTV<T, K> >(
      new TProxTV<T, K>(strength, start, end, positive)));
}

template <class T, class K>
void TProxBinarsity<T, K>::call(const Array<K>& coeffs, T step, Array<K>& out,
                                ulong start, ulong end) {
  if (!is_synchronized) {
    synchronize_proxs();
  }
  for (auto& prox : proxs) {
    ulong start_k = prox->get_start();
    ulong end_k = prox->get_end();
    prox->call(coeffs, step, out, start_k, end_k);
    auto out_block_k = view(out, start_k, end_k);
    T mean_k = out_block_k.sum() / (end_k - start_k);
    for (ulong j = 0; j < end_k - start_k; j++) {
      out_block_k[j] = out_block_k[j] - mean_k;
    }
  }
}

using ProxBinarsityDouble = TProxBinarsity<double, double>;
CEREAL_SPECIALIZE_FOR_ALL_ARCHIVES(ProxBinarsityDouble,
                                   cereal::specialization::member_serialize)
CEREAL_REGISTER_TYPE(ProxBinarsityDouble)

using ProxBinarsityFloat = TProxBinarsity<float, float>;
CEREAL_SPECIALIZE_FOR_ALL_ARCHIVES(ProxBinarsityFloat,
                                   cereal::specialization::member_serialize)
CEREAL_REGISTER_TYPE(ProxBinarsityFloat)

#endif  // LIB_INCLUDE_TICK_PROX_PROX_BINARSITY_H_
