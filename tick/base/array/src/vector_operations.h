#ifndef TICK_BASE_ARRAY_SRC_VECTOR_OPERATIONS_H_
#define TICK_BASE_ARRAY_SRC_VECTOR_OPERATIONS_H_

// License: BSD 3 clause

#include <numeric>

#include "defs.h"
#include "promote.h"

namespace tick {
namespace detail {

template<typename T>
struct vector_operations_unoptimized {
  T dot(const ulong n, const T *x, const T *y) const {
    T result{0};

    for (ulong i = 0; i < n; ++i) {
      result += x[i] * y[i];
    }

    return result;
  }

  tick::promote_t<T> sum(const ulong n, const T *x) const {
    return std::accumulate(x, x + n, tick::promote_t<T>{0});
  }

  void scale(const ulong n, const T alpha, T *x) const {
    for (ulong i = 0; i < n; ++i) {
      x[i] *= alpha;
    }
  }

  void set(const ulong n, const T alpha, T *x) const {
    for (ulong i = 0; i < n; ++i) {
      x[i] = alpha;
    }
  }

  void mult_incr(const ulong n, const T alpha, const T *x, T *y) const {
    for (ulong i = 0; i < n; ++i) {
      y[i] += alpha * x[i];
    }
  }

  void dot_matrix_vector_incr(const ulong m, const ulong n, const T alpha, const T* a, const T* x,
                              const T beta, T* y) const {
    for (ulong i = 0; i < m; ++i) {
      y[i] = beta * y[i];
      for (ulong j = 0; j < n; ++j) {
        y[i] += alpha * a[i * n + j] * x[j];
      }
    }
  }
};

}  // namespace detail
}  // namespace tick

#if !defined(TICK_CBLAS_AVAILABLE) && !defined(TICK_MKL_AVAILABLE)

namespace tick {

template<typename T>
using vector_operations = detail::vector_operations_unoptimized<T>;

}  // namespace tick

#else

#if defined(TICK_MKL_AVAILABLE)

#include <mkl.h>

#elif defined(__APPLE__)

#include <Accelerate/Accelerate.h>

// TODO(svp) Disabling this feature until we find a good way to determine if ATLAS is actually available
// #define XDATA_CATLAS_AVAILABLE

#else

extern "C" {
#include <cblas.h>
}

#endif  // defined(__APPLE__)

namespace tick {

namespace detail {

template<typename T>
struct vector_operations_cblas : vector_operations_unoptimized<T> {};

template<typename T>
struct vector_operations_cblas_base {
  promote_t<T> sum(const ulong n, const T *x) const {
    return vector_operations_unoptimized<T>{}.sum(n, x);
  }

  void set(const ulong n, const T alpha, T *x) const {
    return vector_operations_unoptimized<T>{}.set(n, alpha, x);
  }

  void dot_matrix_vector(const ulong m, const ulong n, const T alpha, const T* a, const T* x, T* y) const {
    for (ulong i = 0; i < m; ++i) {
      y[i] = 0;
      for (ulong j = 0; j < n; ++j) {
        y[i] += alpha * a[i * n + j] * x[j];
      }
    }
  }
};

template<>
struct vector_operations_cblas<float> final : public vector_operations_cblas_base<float> {
  float absolute_sum(const ulong n, const float *x) const {
    return cblas_sasum(n, x, 1);
  }

  float dot(const ulong n, const float *x, const float *y) const {
    return cblas_sdot(n, x, 1, y, 1);
  }

  void scale(const ulong n, const float alpha, float *x) const {
    cblas_sscal(n, alpha, x, 1);
  }

  void mult_incr(const ulong n, const float alpha, const float *x, float *y) const {
    cblas_saxpy(n, alpha, x, 1, y, 1);
  }

#if defined(TICK_CATLAS_AVAILABLE)
  void set(const ulong n, const float alpha, float* x) const override {
      catlas_sset(n, alpha, x, 1);
  }
#endif
};

template<>
struct vector_operations_cblas<double> final : public vector_operations_cblas_base<double> {
  double absolute_sum(const ulong n, const double *x) const {
    return cblas_dasum(n, x, 1);
  }

  double dot(const ulong n, const double *x, const double *y) const {
    return cblas_ddot(n, x, 1, y, 1);
  }

  void scale(const ulong n, const double alpha, double *x) const {
    cblas_dscal(n, alpha, x, 1);
  }

  void mult_incr(const ulong n, const double alpha, const double *x, double *y) const {
    cblas_daxpy(n, alpha, x, 1, y, 1);
  }

#if defined(TICK_CATLAS_AVAILABLE)
  void set(const ulong n, const double alpha, double* x) const {
    catlas_dset(n, alpha, x, 1);
  }
#endif

  void dot_matrix_vector_incr(const ulong m, const ulong n, const double alpha, const double* a, const double* x,
                              const double beta, double* y) const {
    cblas_dgemv(
      CBLAS_ORDER::CblasRowMajor, CBLAS_TRANSPOSE::CblasNoTrans, m, n, alpha, a, n, x, 1, beta, y, 1);
//    for (ulong i = 0; i < m; ++i) {
//      y[i] = beta * z[i];
//      for (ulong j = 0; j < n; ++j) {
//        y[i] += alpha * a[i * n + j] * x[j];
//      }
//    }
  }
};

}  // namespace detail

template<typename T>
using vector_operations = detail::vector_operations_cblas<T>;

}  // namespace tick

#endif  // if !defined(TICK_CBLAS_AVAILABLE)

#endif  // TICK_BASE_ARRAY_SRC_VECTOR_OPERATIONS_H_
