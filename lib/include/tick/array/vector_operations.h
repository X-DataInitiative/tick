#ifndef LIB_INCLUDE_TICK_ARRAY_VECTOR_OPERATIONS_H_
#define LIB_INCLUDE_TICK_ARRAY_VECTOR_OPERATIONS_H_

// License: BSD 3 clause

#include <numeric>

#include "promote.h"
#include "tick/base/defs.h"

namespace tick {
namespace detail {

template <typename T>
struct DLL_PUBLIC vector_operations_unoptimized {
  template <typename K>
  T dot(const ulong n, const T *x, const K *y) const;

  template <typename K = T>
  tick::promote_t<K> sum(const ulong n, const T *x) const;

  template <typename K = T>
  void scale(const ulong n, const K alpha, T *x) const;

  template <typename K>
  void set(const ulong n, const K alpha, T *x) const;

  template <typename K = T>
  void mult_incr(const ulong n, const K alpha, const T *x, T *y) const;
};

}  // namespace detail
}  // namespace tick

#if !defined(TICK_CBLAS_AVAILABLE)

namespace tick {

template <typename T>
using vector_operations = detail::vector_operations_unoptimized<T>;

}  // namespace tick

#else  // if defined(TICK_CBLAS_AVAILABLE)

// Find available blas distribution
#if defined(TICK_USE_MKL)

#include <mkl.h>

#elif defined(__APPLE__)

#include <Accelerate/Accelerate.h>

// TODO(svp) Disabling this feature until we find a good way to determine if
// ATLAS is actually available #define XDATA_CATLAS_AVAILABLE

#else

extern "C" {
#include <cblas.h>
}

#endif  // defined(__APPLE__)

namespace tick {

namespace detail {

template <typename T>
struct vector_operations_cblas : vector_operations_unoptimized<T> {};

template <typename T>
struct vector_operations_cblas_base {
  template <typename K = T>
  promote_t<T> sum(const ulong n, const T *x) const {
    return vector_operations_unoptimized<T>{}.template sum<K>(n, x);
  }

  template <typename K>
  void set(const ulong n, const K alpha, T *x) const {
    return vector_operations_unoptimized<T>{}.template set<K>(n, alpha, x);
  }
};

template <>
struct vector_operations_cblas<float> final
    : public vector_operations_cblas_base<float> {
  float absolute_sum(const ulong n, const float *x) const {
    return cblas_sasum(n, x, 1);
  }

  template <typename K>  // see cblas.cpp
  float dot(const ulong n, const float *x, const K *y) const;

  template <typename K>
  void scale(const ulong n, const K alpha, float *x) const {
    cblas_sscal(n, alpha, x, 1);
  }

  template <typename K>  // see cblas.cpp
  void mult_incr(const ulong n, const K alpha, const float *x, float *y) const;

#if defined(TICK_CATLAS_AVAILABLE)
  template <typename K>
  void set(const ulong n, const K alpha, float *x) const override {
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

  template <typename K>  // see cblas.cpp
  double dot(const ulong n, const double *x, const K *y) const;

  template <typename K>
  void scale(const ulong n, const K alpha, double *x) const {
    cblas_dscal(n, alpha, x, 1);
  }

  template <typename K>  // see cblas.cpp
  void mult_incr(const ulong n, const K alpha, const double *x,
                 double *y) const;

#if defined(TICK_CATLAS_AVAILABLE)
  template <typename K>
  void set(const ulong n, const K alpha, double *x) const {
    catlas_dset(n, alpha, x, 1);
  }
#endif
};

}  // namespace detail

template <typename T>
using vector_operations = detail::vector_operations_cblas<T>;

}  // namespace tick

#endif  // if !defined(TICK_CBLAS_AVAILABLE)

#endif  // LIB_INCLUDE_TICK_ARRAY_VECTOR_OPERATIONS_H_
