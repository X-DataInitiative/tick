
#include "tick/array/array.h"

namespace tick {
namespace detail {

template <typename T>
template <typename K>
T vector_operations_unoptimized<T>::dot(const uint64_t n, const T *x,
                                        const K *y) const {
  T result{0};

  for (uint64_t i = 0; i < n; ++i) {
    result += x[i] * y[i];
  }

  return result;
}

template <>
template <>
double vector_operations_unoptimized<double>::dot<std::atomic<double>>(
    const uint64_t n, const double *x, const std::atomic<double> *y) const {
  double result{0};

  for (uint64_t i = 0; i < n; ++i) {
    result += x[i] * y[i].load();
  }

  return result;
}

template <>
template <>
float vector_operations_unoptimized<float>::dot<std::atomic<float>>(
    const uint64_t n, const float *x, const std::atomic<float> *y) const {
  float result{0};

  for (uint64_t i = 0; i < n; ++i) {
    result += x[i] * y[i].load();
  }

  return result;
}

}  // namespace detail
}  // namespace tick

template DLL_PUBLIC int
tick::detail::vector_operations_unoptimized<int>::dot<int>(uint64_t,
                                                           int const *,
                                                           int const *) const;

template DLL_PUBLIC unsigned int
tick::detail::vector_operations_unoptimized<unsigned int>::dot<unsigned int>(
    uint64_t, const unsigned int *, const unsigned int *) const;

template DLL_PUBLIC uint64_t
tick::detail::vector_operations_unoptimized<uint64_t>::dot<uint64_t>(
    uint64_t, const uint64_t *, const uint64_t *) const;

template DLL_PUBLIC unsigned short tick::detail::vector_operations_unoptimized<
    unsigned short>::dot<unsigned short>(uint64_t, const unsigned short *,
                                         const unsigned short *) const;

template DLL_PUBLIC float tick::detail::vector_operations_unoptimized<
    float>::dot<float>(uint64_t, float const *, float const *) const;

template DLL_PUBLIC short tick::detail::vector_operations_unoptimized<
    short>::dot<short>(uint64_t, short const *, short const *) const;

template DLL_PUBLIC double tick::detail::vector_operations_unoptimized<
    double>::dot<double>(uint64_t, double const *, double const *) const;

#if defined(_MSC_VER)
template DLL_PUBLIC __int64 tick::detail::vector_operations_unoptimized<__int64>::dot<
    __int64>(unsigned __int64, __int64 const *, __int64 const *) const;
#else

template DLL_PUBLIC long long tick::detail::vector_operations_unoptimized<long long>::dot<
    long long>(uint64_t, const long long *, const long long *) const;

#endif

template DLL_PUBLIC long tick::detail::vector_operations_unoptimized<long>::dot<long>(
    ulong, long const *, long const *) const;
