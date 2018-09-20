#ifndef LIB_INCLUDE_TICK_ARRAY_VECTOR_OPERATIONS_H_
#define LIB_INCLUDE_TICK_ARRAY_VECTOR_OPERATIONS_H_

// License: BSD 3 clause

#include <numeric>

#include <algorithm>
#include <atomic>
#include <type_traits>

#include "promote.h"
#include "tick/base/defs.h"

namespace tick {
namespace detail {

template <typename T>
struct DLL_PUBLIC vector_operations_unoptimized {
  template <typename K>
  tick::promote_t<K> sum(const ulong n, const T *x) const;

  template <typename K>
  typename std::enable_if<std::is_same<T, std::atomic<K>>::value, T>::type dot(
      const ulong n, const T *x, const K *y) const;

  template <typename K>
  typename std::enable_if<std::is_same<T, std::atomic<K>>::value, K>::type dot(
      const ulong n, const K *x, const T *y) const;

  template <typename K>
  typename std::enable_if<!std::is_same<T, std::atomic<K>>::value, T>::type dot(
      const ulong n, const T *x, const K *y) const;

  template <typename K>
  typename std::enable_if<std::is_same<T, std::atomic<K>>::value>::type scale(
      const ulong n, const K alpha, T *x) const;

  template <typename K>
  typename std::enable_if<!std::is_same<T, std::atomic<K>>::value>::type scale(
      const ulong n, const K alpha, T *x) const;

  template <typename K>
  typename std::enable_if<std::is_same<T, std::atomic<K>>::value>::type set(
      const ulong n, const K alpha, T *x) const;

  template <typename K>
  typename std::enable_if<!std::is_same<T, std::atomic<K>>::value>::type set(
      const ulong n, const K alpha, T *x) const;

  template <typename K, typename Y>
  typename std::enable_if<std::is_same<T, std::atomic<K>>::value &&
                          !std::is_same<Y, std::atomic<K>>::value>::type
  mult_incr(const uint64_t n, const K alpha, const Y *x, T *y) const;

  template <typename K, typename Y>
  typename std::enable_if<std::is_same<Y, std::atomic<K>>::value &&
                          !std::is_same<T, std::atomic<K>>::value>::type
  mult_incr(const uint64_t n, const K alpha, const Y *x, T *y) const;

  template <typename K, typename Y>
  typename std::enable_if<std::is_same<T, std::atomic<K>>::value &&
                          std::is_same<Y, std::atomic<K>>::value>::type
  mult_incr(const uint64_t n, const K alpha, const Y *x, T *y) const;

  template <typename K, typename Y>
  typename std::enable_if<std::is_same<T, K>::value && std::is_same<Y, K>::value>::type mult_incr(
      const uint64_t n, const K alpha, const Y *x, T *y) const;

  template <typename K>
  std::vector<T> batch_dot(size_t n_threads, const ulong b, const ulong n, T *x, T **const y) const;

  template <typename K>
  void batch_multi_incr(size_t n_threads, const ulong b, const ulong n, const K *alpha, T **const x,
                        T *y) const;

  template <typename K>
  typename std::enable_if<std::is_same<T, std::atomic<K>>::value>::type dot_matrix_vector_incr(
      const ulong m, const ulong n, const T alpha, const T *a, const T *x, const T beta,
      T *y) const;

  template <typename K>
  typename std::enable_if<!std::is_same<T, std::atomic<K>>::value>::type dot_matrix_vector_incr(
      const ulong m, const ulong n, const T alpha, const T *a, const T *x, const T beta,
      T *y) const;

  template <typename K>
  typename std::enable_if<std::is_same<T, std::atomic<K>>::value>::type dot_matrix_vector(
      const ulong m, const ulong n, const K alpha, const T *a, const T *x, T *y) const;

  template <typename K>
  typename std::enable_if<!std::is_same<T, std::atomic<K>>::value>::type dot_matrix_vector(
      const ulong m, const ulong n, const K alpha, const T *a, const T *x, T *y) const;
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

#include "tick/array/vector/cblas.h"

#endif  // if !defined(TICK_CBLAS_AVAILABLE)

#include "tick/array/vector/unoptimized.h"

#endif  // LIB_INCLUDE_TICK_ARRAY_VECTOR_OPERATIONS_H_
