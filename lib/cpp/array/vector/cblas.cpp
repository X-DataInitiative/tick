

#if defined(TICK_CBLAS_AVAILABLE)

#include "tick/array/array.h"

namespace tick {
namespace detail {

template <>
void vector_operations_cblas<float>::mult_incr<float>(const ulong n,
                                                      const float alpha,
                                                      const float *x,
                                                      float *y) const {
  cblas_saxpy(n, alpha, x, 1, y, 1);
}
template <>
void vector_operations_cblas<float>::mult_incr<std::atomic<float> >(
    const ulong n, const std::atomic<float> alpha, const float *x,
    float *y) const {
  cblas_saxpy(n, alpha.load(), x, 1, y, 1);
}

template <>
void vector_operations_cblas<double>::mult_incr<double>(const ulong n,
                                                        const double alpha,
                                                        const double *x,
                                                        double *y) const {
  cblas_daxpy(n, alpha, x, 1, y, 1);
}

template <>
void vector_operations_cblas<double>::mult_incr<std::atomic<double> >(
    const ulong n, const std::atomic<double> alpha, const double *x,
    double *y) const {
  cblas_daxpy(n, alpha.load(), x, 1, y, 1);
}

template <>
void tick::detail::vector_operations_cblas<float>::mult_incr<double>(
    const ulong n, const double alpha, const float *x, float *y) const {
  vector_operations_unoptimized<float>{}.template mult_incr(n, alpha, x, y);
}

template <>
void tick::detail::vector_operations_cblas<float>::mult_incr<int>(
    const ulong n, const int alpha, const float *x, float *y) const {
  vector_operations_unoptimized<float>{}.template mult_incr<int>(n, alpha, x,
                                                                 y);
}

template <>
void tick::detail::vector_operations_cblas<double>::mult_incr<int>(
    const ulong n, const int alpha, const double *x, double *y) const {
  vector_operations_unoptimized<double>{}.template mult_incr<int>(n, alpha, x,
                                                                  y);
}

template <>
float vector_operations_cblas<float>::dot(const ulong n, const float *x,
                                          const float *y) const {
  return cblas_sdot(n, x, 1, y, 1);
}

template <>
float vector_operations_cblas<float>::dot<std::atomic<float> >(
    const ulong n, const float *x, const std::atomic<float> *y) const {
  // return cblas_sdot(n, x, 1, y, 1);

  return vector_operations_unoptimized<float>{}
      .template dot<std::atomic<float> >(n, x, y);
}

template <>
double vector_operations_cblas<double>::dot(const ulong n, const double *x,
                                            const double *y) const {
  return cblas_ddot(n, x, 1, y, 1);
}

template <>
double vector_operations_cblas<double>::dot<std::atomic<double> >(
    const ulong n, const double *x, const std::atomic<double> *y) const {
  // return cblas_ddot(n, x, 1, y, 1);
  return vector_operations_unoptimized<double>{}
      .template dot<std::atomic<double> >(n, x, y);
}

template <>
void vector_operations_cblas<float>::scale(const ulong n, const float alpha,
                                           float *x) const {
  cblas_sscal(n, alpha, x, 1);
}
template <>
void vector_operations_cblas<float>::scale<std::atomic<float> >(
    const ulong n, const std::atomic<float> alpha, float *x) const {
  cblas_sscal(n, alpha.load(), x, 1);
}

template <>
void vector_operations_cblas<double>::scale(const ulong n, const double alpha,
                                            double *x) const {
  cblas_dscal(n, alpha, x, 1);
}
template <>
void vector_operations_cblas<double>::scale<std::atomic<double> >(
    const ulong n, const std::atomic<double> alpha, double *x) const {
  cblas_dscal(n, alpha.load(), x, 1);
}

#if defined(TICK_CATLAS_AVAILABLE)
template <>
void vector_operations_cblas<float>::scale(const ulong n, const float alpha,
                                           float *x) const {
  catlas_sset(n, alpha, x, 1);
}
template <>
void vector_operations_cblas<double>::scale(const ulong n, const double alpha,
                                            double *x) const {
  catlas_dset(n, alpha, x, 1);
}
#endif

}  // end namespace detail
}  // end namespace tick

#endif  // defined(TICK_CBLAS_AVAILABLE)
