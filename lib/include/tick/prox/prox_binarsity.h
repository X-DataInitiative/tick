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

 protected:
  std::unique_ptr<TProx<T> > build_prox(T strength, ulong start, ulong end,
                                        bool positive) override;

 public:
  TProxBinarsity(T strength, SArrayULongPtr blocks_start,
                 SArrayULongPtr blocks_length, bool positive);

  TProxBinarsity(T strength, SArrayULongPtr blocks_start,
                 SArrayULongPtr blocks_length, ulong start, ulong end,
                 bool positive);

  TProxBinarsity() = delete;
  TProxBinarsity(const TProxBinarsity& other) = delete;
  TProxBinarsity(TProxBinarsity&& other) = delete;
  TProxBinarsity& operator=(const TProxBinarsity& other) = delete;
  TProxBinarsity& operator=(TProxBinarsity&& other) = delete;

  std::string get_class_name() const;

  void call(const Array<T>& coeffs, T step, Array<T>& out, ulong start,
            ulong end);
};

using ProxBinarsity = TProxBinarsity<double>;

using ProxBinarsityDouble = TProxBinarsity<double>;
using ProxBinarsityFloat = TProxBinarsity<double>;

#endif  // LIB_INCLUDE_TICK_PROX_PROX_BINARSITY_H_
