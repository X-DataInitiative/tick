
#include "tick/array/array.h"


namespace tick {
namespace detail {

template <typename T>
template <typename K>
void vector_operations_unoptimized<T>::mult_incr(const uint64_t n,
                                                 const K alpha, const T *x,
                                                 T *y) const {
  for (uint64_t i = 0; i < n; ++i) {
    y[i] += alpha * x[i];
  }
}

template <>
template <>
void vector_operations_unoptimized<std::atomic<double>>::mult_incr<double>(
    const uint64_t n, const double alpha, const std::atomic<double> *x,
    std::atomic<double> *y) const {
  for (uint64_t i = 0; i < n; ++i) {
    double y_i = y[i].load();
    y_i += alpha * x[i];
    y[i].store(y_i);
  }
}
template <>
template <>
void vector_operations_unoptimized<std::atomic<float>>::mult_incr<float>(
    const uint64_t n, const float alpha, const std::atomic<float> *x,
    std::atomic<float> *y) const {
  for (uint64_t i = 0; i < n; ++i) {
    float y_i = y[i].load();
    y_i += alpha * x[i];
    y[i].store(y_i);
  }
}

}  // namespace detail
}  // namespace tick

template DLL_PUBLIC void
tick::detail::vector_operations_unoptimized<int64_t>::mult_incr<int64_t>(
    uint64_t, int64_t, int64_t const *, int64_t *) const;

template DLL_PUBLIC void
tick::detail::vector_operations_unoptimized<int>::mult_incr<int>(uint64_t, int,
                                                                 int const *,
                                                                 int *) const;

template DLL_PUBLIC void tick::detail::vector_operations_unoptimized<
    unsigned int>::mult_incr<unsigned int>(uint64_t, unsigned int,
                                           unsigned int const *,
                                           unsigned int *) const;
template DLL_PUBLIC void
tick::detail::vector_operations_unoptimized<uint64_t>::mult_incr<uint64_t>(
    uint64_t, uint64_t, uint64_t const *, uint64_t *) const;

template DLL_PUBLIC void tick::detail::vector_operations_unoptimized<
    unsigned short>::mult_incr<unsigned short>(uint64_t, unsigned short,
                                               unsigned short const *,
                                               unsigned short *) const;

template DLL_PUBLIC void tick::detail::vector_operations_unoptimized<
    float>::mult_incr<int>(uint64_t, int, float const *, float *) const;

template DLL_PUBLIC void tick::detail::vector_operations_unoptimized<
    float>::mult_incr<float>(uint64_t, float, float const *, float *) const;

template DLL_PUBLIC void tick::detail::vector_operations_unoptimized<
    float>::mult_incr<double>(uint64_t, double, float const *, float *) const;

template DLL_PUBLIC void tick::detail::vector_operations_unoptimized<
    short>::mult_incr<short>(uint64_t, short, short const *, short *) const;

template DLL_PUBLIC void tick::detail::vector_operations_unoptimized<
    double>::mult_incr<int>(uint64_t, int, double const *, double *) const;

template DLL_PUBLIC void
tick::detail::vector_operations_unoptimized<double>::mult_incr<double>(
    uint64_t, double, double const *, double *) const;

#if defined(_MSC_VER)

template DLL_PUBLIC void
tick::detail::vector_operations_unoptimized<std::atomic<float>>::set<int>(
    unsigned __int64, int, std::atomic<float> *) const;

template DLL_PUBLIC void tick::detail::vector_operations_unoptimized<
    std::atomic<float>>::mult_incr<float>(unsigned __int64, const float,
                                          const std::atomic<float> *,
                                          std::atomic<float> *) const;

template DLL_PUBLIC void tick::detail::vector_operations_unoptimized<
    std::atomic<double>>::mult_incr<double>(unsigned __int64, const double,
                                            const std::atomic<double> *,
                                            std::atomic<double> *) const;

#endif
