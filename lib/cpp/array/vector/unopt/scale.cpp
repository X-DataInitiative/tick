
#include "tick/array/array.h"

namespace tick {
namespace detail {

template <typename T>
template <typename K>
void vector_operations_unoptimized<T>::scale(const uint64_t n, const K alpha,
                                             T *x) const {
  for (uint64_t i = 0; i < n; ++i) {
    x[i] *= alpha;
  }
}
template <>
template <>
void vector_operations_unoptimized<std::atomic<double>>::scale<double>(
    const uint64_t n, const double alpha, std::atomic<double> *x) const {
  for (uint64_t i = 0; i < n; ++i) {
    double x_i = x[i].load();
    x_i *= alpha;
    x[i].store(x_i);
  }
}
template <>
template <>
void vector_operations_unoptimized<std::atomic<float>>::scale<float>(
    const uint64_t n, const float alpha, std::atomic<float> *x) const {
  for (uint64_t i = 0; i < n; ++i) {
    float x_i = x[i].load();
    x_i *= alpha;
    x[i].store(x_i);
  }
}

}  // namespace detail
}  // namespace tick

template DLL_PUBLIC void tick::detail::vector_operations_unoptimized<
    double>::scale<double>(const uint64_t, const double, double *) const;

template DLL_PUBLIC void
tick::detail::vector_operations_unoptimized<int>::scale<int>(const uint64_t,
                                                             const int,
                                                             int *) const;
template DLL_PUBLIC void
tick::detail::vector_operations_unoptimized<short>::scale<short>(const uint64_t,
                                                                 const short,
                                                                 short *) const;

template DLL_PUBLIC void
tick::detail::vector_operations_unoptimized<float>::scale<float>(const uint64_t,
                                                                 const float,
                                                                 float *) const;

template DLL_PUBLIC void tick::detail::vector_operations_unoptimized<
    float>::scale<double>(const uint64_t, const double, float *) const;

template DLL_PUBLIC void
tick::detail::vector_operations_unoptimized<unsigned int>::scale<unsigned int>(
    const uint64_t, const unsigned int, unsigned int *) const;

template DLL_PUBLIC void
tick::detail::vector_operations_unoptimized<uint64_t>::scale<uint64_t>(
    const uint64_t, const uint64_t, uint64_t *) const;

template DLL_PUBLIC void tick::detail::vector_operations_unoptimized<
    unsigned short>::scale<unsigned short>(const uint64_t, const unsigned short,
                                           unsigned short *) const;

#if defined(_MSC_VER)

template DLL_PUBLIC void
tick::detail::vector_operations_unoptimized<std::atomic<float>>::scale<float>(
    const uint64_t n, const float, std::atomic<float> *x) const;

template DLL_PUBLIC void
tick::detail::vector_operations_unoptimized<std::atomic<double>>::scale<double>(
    const uint64_t n, const double, std::atomic<double> *x) const;

template DLL_PUBLIC void tick::detail::vector_operations_unoptimized<
    unsigned long>::scale<unsigned long>(unsigned __int64, unsigned long,
                                         unsigned long *) const;

template DLL_PUBLIC void
tick::detail::vector_operations_unoptimized<long>::scale<long>(unsigned __int64,
                                                               long,
                                                               long *) const;
#else

template void tick::detail::vector_operations_unoptimized<long>::scale<long>(
    const uint64_t n, const long, long *) const;

#endif

#if defined(__APPLE__)

template void tick::detail::vector_operations_unoptimized<unsigned long>::scale<
    unsigned long>(const unsigned long long, const unsigned long,
                   unsigned long *) const;
#endif

template DLL_PUBLIC void
tick::detail::vector_operations_unoptimized<long long>::scale<long long>(
    const uint64_t, const long long, long long *) const;
