

#ifndef LIB_INCLUDE_TICK_ARRAY_VECTOR_OPS_BLAS_H_
#define LIB_INCLUDE_TICK_ARRAY_VECTOR_OPS_BLAS_H_

#include "tick/array/vector/ops_unoptimized.h"

namespace tick {
namespace detail {

template <typename T>
struct vector_operations_cblas : vector_operations_unoptimized<T> {};

template <typename T>
struct vector_operations_cblas_base : vector_operations_unoptimized<T> {
  template <typename K>
  promote_t<T> sum(const ulong n, const T *x) const {
    return vector_operations_unoptimized<T>{}.template sum<K>(n, x);
  }

  template <typename K>
  void set(const ulong n, const K alpha, T *x) const {
    vector_operations_unoptimized<T>{}.set(n, alpha, x);
  }

  template <typename K>
  typename std::enable_if<std::is_same<T, std::atomic<K>>::value>::type scale(const ulong n,
                                                                              const K alpha,
                                                                              T *x) const {
    vector_operations_unoptimized<T>{}.scale(n, alpha, x);
  }

  template <typename K = T>
  typename std::enable_if<std::is_same<T, std::atomic<K>>::value, T>::type
  dot(const ulong n, const T *x, const K *y) const {
    return vector_operations_unoptimized<float>{}.dot(n, x, y);
  }

  template <typename K = T>
  typename std::enable_if<std::is_same<T, std::atomic<K>>::value, K>::type
  dot(const ulong n, const K *x, const T *y) const {
    return vector_operations_unoptimized<T>{}.dot(n, x, y);
  }
};

template <>
struct vector_operations_cblas<float> final
    : public vector_operations_cblas_base<float> {
  float absolute_sum(const ulong n, const float *x) const {
    return cblas_sasum(n, x, 1);
  }
  float dot(const ulong n, const float *x, const float *y) const {
    return cblas_sdot(n, x, 1, y, 1);
  }

  float dot(const ulong n, const float *x, const std::atomic<float> *y) const {
    return vector_operations_unoptimized<float>{}.dot(n, x, y);
  }
  void scale(const ulong n, const std::atomic<float> alpha, float *x) const {
    cblas_sscal(n, alpha.load(), x, 1);
  }
  void scale(const ulong n, float alpha, float *x) const {
    cblas_sscal(n, alpha, x, 1);
  }
  void mult_incr(const uint64_t n, const float alpha, const std::atomic<float> *x, float *y) const {
    vector_operations_unoptimized<float>{}.mult_incr(n, alpha, x, y);
  }
  void mult_incr(const uint64_t n, const float alpha, const float *x, float *y) const {
    cblas_saxpy(n, alpha, x, 1, y, 1);
  }

#if defined(TICK_USE_CATLAS_)
  void set(const ulong n, const float alpha, float *x) const {
    catlas_sset(n, alpha, x, 1);
  }
#endif
};

template <>
struct vector_operations_cblas<double> final
    : public vector_operations_cblas_base<double> {
  double absolute_sum(const ulong n, const double *x) const {
    return cblas_dasum(n, x, 1);
  }
  double dot(const ulong n, const double *x, const double *y) const {
    return cblas_ddot(n, x, 1, y, 1);
  }

  double dot(const ulong n, const double *x, const std::atomic<double> *y) const {
    return vector_operations_unoptimized<double>{}.dot(n, x, y);
  }
  void mult_incr(const uint64_t n, const double alpha, const double *x, double *y) const {
    cblas_daxpy(n, alpha, x, 1, y, 1);
  }
  void mult_incr(const uint64_t n, const double alpha, const std::atomic<double> *x, double *y) const {
    vector_operations_unoptimized<double>{}.mult_incr(n, alpha, x, y);
  }
  void scale(const ulong n, const std::atomic<double> alpha, double *x) const {
    cblas_dscal(n, alpha.load(), x, 1);
  }
  void scale(const ulong n, double alpha, double *x) const {
    cblas_dscal(n, alpha, x, 1);
  }
#if defined(TICK_USE_CATLAS_)
  void set(const ulong n, const double alpha, double *x) const {
    catlas_sset(n, alpha, x, 1);
  }
#endif
};


}  // namespace detail
}  // namespace tick

#endif  // LIB_INCLUDE_TICK_ARRAY_VECTOR_OPS_BLAS_H_
