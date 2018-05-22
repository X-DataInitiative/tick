
#include "tick/array/array.h"

namespace tick {
namespace detail {

template <typename T>
template <typename K>
tick::promote_t<K> vector_operations_unoptimized<T>::sum(const ulong n,
                                                         const T *x) const {
  return std::accumulate(x, x + n, tick::promote_t<K>{0});
}

template <>
template <>
tick::promote_t<float>
vector_operations_unoptimized<std::atomic<float> >::sum<float>(
    const ulong n, const std::atomic<float> *x) const {
  return std::accumulate(x, x + n, tick::promote_t<float>{0});
}

template <>
template <>
tick::promote_t<double>
vector_operations_unoptimized<std::atomic<double> >::sum<double>(
    const ulong n, const std::atomic<double> *x) const {
  return std::accumulate(x, x + n, tick::promote_t<double>{0});
}

}  // namespace detail
}  // namespace tick

template DLL_PUBLIC tick::promote_t<float>
tick::detail::vector_operations_unoptimized<float>::sum<float>(
    const ulong, const float *) const;

template DLL_PUBLIC tick::promote_t<double>
tick::detail::vector_operations_unoptimized<double>::sum<double>(
    const ulong, const double *) const;

template DLL_PUBLIC tick::promote_t<short>
tick::detail::vector_operations_unoptimized<short>::sum<short>(
    const ulong, const short *) const;

template DLL_PUBLIC tick::promote_t<unsigned short>
tick::detail::vector_operations_unoptimized<unsigned short>::sum<
    unsigned short>(const ulong, const unsigned short *) const;

template DLL_PUBLIC tick::promote_t<int>
tick::detail::vector_operations_unoptimized<int>::sum<int>(const ulong,
                                                           const int *) const;

template DLL_PUBLIC tick::promote_t<unsigned int>
tick::detail::vector_operations_unoptimized<unsigned int>::sum<unsigned int>(
    const ulong, const unsigned int *) const;

template DLL_PUBLIC tick::promote_t<int64_t>
tick::detail::vector_operations_unoptimized<int64_t>::sum<int64_t>(
    uint64_t, int64_t const *) const;

template DLL_PUBLIC tick::promote_t<uint64_t>
tick::detail::vector_operations_unoptimized<uint64_t>::sum<
    uint64_t>(uint64_t, uint64_t const *) const;

#if defined(_MSC_VER)

template DLL_PUBLIC tick::promote_t<float>
tick::detail::vector_operations_unoptimized<std::atomic<float> >::sum<float>(
    const ulong n, const std::atomic<float> *x) const;

template DLL_PUBLIC tick::promote_t<double>
tick::detail::vector_operations_unoptimized<std::atomic<double> >::sum<double>(
    const ulong n, const std::atomic<double> *x) const;

#endif
