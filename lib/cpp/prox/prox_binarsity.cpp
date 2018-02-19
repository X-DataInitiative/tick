// License: BSD 3 clause

#include "tick/prox/prox_binarsity.h"
#include "tick/prox/prox_tv.h"

template <class T>
TProxBinarsity<T>::TProxBinarsity(T strength, SArrayULongPtr blocks_start,
                                  SArrayULongPtr blocks_length, bool positive)
    : TProxWithGroups<T>(strength, blocks_start, blocks_length, positive) {}

template <class T>
TProxBinarsity<T>::TProxBinarsity(T strength, SArrayULongPtr blocks_start,
                                  SArrayULongPtr blocks_length, ulong start,
                                  ulong end, bool positive)
    : TProxWithGroups<T>(strength, blocks_start, blocks_length, start, end,
                         positive) {}

template <class T>
std::unique_ptr<TProx<T> > TProxBinarsity<T>::build_prox(T strength,
                                                         ulong start, ulong end,
                                                         bool positive) {
  return std::move(std::unique_ptr<TProxTV<T> >(
      new TProxTV<T>(strength, start, end, positive)));
}

template <class T>
std::string TProxBinarsity<T>::get_class_name() const {
  return "TProxBinarsity";
}

template <class T>
void TProxBinarsity<T>::call(const Array<T> &coeffs, T step, Array<T> &out,
                             ulong start, ulong end) {
  if (!is_synchronized) {
    synchronize_proxs();
  }
  for (auto &prox : proxs) {
    ulong start_k = prox->get_start();
    ulong end_k = prox->get_end();
    prox->call(coeffs, step, out, start_k, end_k);
    Array<T> out_block_k = view(out, start_k, end_k);
    T mean_k = out_block_k.sum() / (end_k - start_k);
    for (ulong j = 0; j < end_k - start_k; j++) {
      out_block_k[j] -= mean_k;
    }
  }
}

template class DLL_PUBLIC TProxBinarsity<double>;
template class DLL_PUBLIC TProxBinarsity<float>;
