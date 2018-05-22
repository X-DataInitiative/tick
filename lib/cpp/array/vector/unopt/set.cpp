
#include "tick/array/array.h"

namespace tick {
namespace detail {

template <typename T>
template <typename K>
void vector_operations_unoptimized<T>::set(const uint64_t n, const K alpha,
                                           T *x) const {
  for (uint64_t i = 0; i < n; ++i) {
    x[i] = alpha;
  }
}
template <>
template <>
void vector_operations_unoptimized<std::atomic<double> >::set<double>(
    const uint64_t n, const double alpha, std::atomic<double> *x) const {
  for (uint64_t i = 0; i < n; ++i) {
    x[i].store(alpha);
  }
}
template <>
template <>
void vector_operations_unoptimized<std::atomic<float> >::set<float>(
    const uint64_t n, const float alpha, std::atomic<float> *x) const {
  for (uint64_t i = 0; i < n; ++i) {
    x[i].store(alpha);
  }
}
template <>
template <>
void vector_operations_unoptimized<std::atomic<float> >::set<int>(
    const uint64_t n, const int alpha, std::atomic<float> *x) const {
  for (uint64_t i = 0; i < n; ++i) {
    x[i].store(alpha);
  }
}

}  // namespace detail
}  // namespace tick

template DLL_PUBLIC void
tick::detail::vector_operations_unoptimized<int>::set<int>(const uint64_t,
                                                           const int,
                                                           int *) const;

template DLL_PUBLIC void tick::detail::vector_operations_unoptimized<
    int64_t>::set<int64_t>(const uint64_t, const int64_t, int64_t *) const;

template DLL_PUBLIC void tick::detail::vector_operations_unoptimized<
    uint64_t>::set<uint64_t>(const uint64_t, const uint64_t, uint64_t *) const;

template DLL_PUBLIC void
tick::detail::vector_operations_unoptimized<short>::set<short>(const uint64_t,
                                                               const short,
                                                               short *) const;
template DLL_PUBLIC void
tick::detail::vector_operations_unoptimized<float>::set<float>(const uint64_t,
                                                               const float,
                                                               float *) const;
template DLL_PUBLIC void
tick::detail::vector_operations_unoptimized<double>::set<int>(const uint64_t,
                                                              const int,
                                                              double *) const;

template DLL_PUBLIC void
tick::detail::vector_operations_unoptimized<unsigned int>::set<unsigned int>(
    const uint64_t, const unsigned int, unsigned int *) const;

template DLL_PUBLIC void tick::detail::vector_operations_unoptimized<
    unsigned short>::set<unsigned short>(const uint64_t, const unsigned short,
                                         unsigned short *) const;

template DLL_PUBLIC void tick::detail::vector_operations_unoptimized<
    uint64_t>::set<int>(const uint64_t, const int, uint64_t *) const;

template DLL_PUBLIC void
tick::detail::vector_operations_unoptimized<int64_t>::set<int>(const uint64_t,
                                                               const int,
                                                               int64_t *) const;
template DLL_PUBLIC void tick::detail::vector_operations_unoptimized<
    unsigned int>::set<int>(const uint64_t, int, unsigned int *) const;

template DLL_PUBLIC void
tick::detail::vector_operations_unoptimized<short>::set<int>(const uint64_t,
                                                             int,
                                                             short *) const;
template DLL_PUBLIC void
tick::detail::vector_operations_unoptimized<unsigned short>::set<int>(
    const uint64_t, const int, unsigned short *) const;

template DLL_PUBLIC void
tick::detail::vector_operations_unoptimized<float>::set<int>(const uint64_t,
                                                             const int,
                                                             float *) const;
template DLL_PUBLIC void
tick::detail::vector_operations_unoptimized<short>::set<double>(const uint64_t,
                                                                const double,
                                                                short *) const;
template DLL_PUBLIC void
tick::detail::vector_operations_unoptimized<int>::set<double>(const uint64_t,
                                                              const double,
                                                              int *) const;

template DLL_PUBLIC void tick::detail::vector_operations_unoptimized<
    int64_t>::set<double>(const uint64_t, const double, int64_t *) const;

template DLL_PUBLIC void
tick::detail::vector_operations_unoptimized<float>::set<double>(const uint64_t,
                                                                const double,
                                                                float *) const;
template DLL_PUBLIC void
tick::detail::vector_operations_unoptimized<unsigned int>::set<double>(
    const uint64_t, const double, unsigned int *) const;

template DLL_PUBLIC void tick::detail::vector_operations_unoptimized<
    uint64_t>::set<double>(const uint64_t, const double, uint64_t *) const;

template DLL_PUBLIC void
tick::detail::vector_operations_unoptimized<unsigned short>::set<double>(
    const uint64_t, const double, unsigned short *) const;

template DLL_PUBLIC void tick::detail::vector_operations_unoptimized<
    float>::set<uint64_t>(const uint64_t, const uint64_t, float *) const;

template DLL_PUBLIC void tick::detail::vector_operations_unoptimized<
    double>::set<uint64_t>(const uint64_t, const uint64_t, double *) const;

template DLL_PUBLIC void
tick::detail::vector_operations_unoptimized<unsigned int>::set<uint64_t>(
    const uint64_t, const uint64_t, unsigned int *) const;

template DLL_PUBLIC void tick::detail::vector_operations_unoptimized<
    short>::set<uint64_t>(const uint64_t, const uint64_t, short *) const;

template DLL_PUBLIC void
tick::detail::vector_operations_unoptimized<unsigned short>::set<uint64_t>(
    const uint64_t, const uint64_t, unsigned short *) const;

template DLL_PUBLIC void
tick::detail::vector_operations_unoptimized<int>::set<uint64_t>(const uint64_t,
                                                                const uint64_t,
                                                                int *) const;

template DLL_PUBLIC void tick::detail::vector_operations_unoptimized<
    std::atomic<double> >::set<int>(uint64_t, int, std::atomic<double> *) const;

#if defined(_MSC_VER)
template DLL_PUBLIC void
tick::detail::vector_operations_unoptimized<int64_t>::set<int>(const uint64_t,
                                                               const int,
                                                               int64_t *) const;

template DLL_PUBLIC void tick::detail::vector_operations_unoptimized<
    double>::set<double>(const unsigned __int64, const double, double *) const;

template DLL_PUBLIC void
tick::detail::vector_operations_unoptimized<std::atomic<float> >::set<float>(unsigned __int64, float, std::atomic<float> *) const;

template DLL_PUBLIC void
tick::detail::vector_operations_unoptimized<std::atomic<double> >::set<double>(unsigned __int64, double, std::atomic<double> *) const;

template DLL_PUBLIC void
tick::detail::vector_operations_unoptimized<std::atomic<float> >::set<int>(unsigned __int64, int, std::atomic<float> *) const;
#else

template DLL_PUBLIC void tick::detail::vector_operations_unoptimized<
    double>::set<double>(const uint64_t, const double, double *) const;

#endif

template DLL_PUBLIC void tick::detail::vector_operations_unoptimized<
    int64_t>::set<uint64_t>(const uint64_t, const uint64_t, int64_t *) const;
