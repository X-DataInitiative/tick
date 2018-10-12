
#ifndef LIB_INCLUDE_TICK_ARRAY_VECTOR_OPS_UNOPTIMIZED_IMPL_H_
#define LIB_INCLUDE_TICK_ARRAY_VECTOR_OPS_UNOPTIMIZED_IMPL_H_

// This macro ensures that the corresponding optimized blas function is used if available
#if (DEBUG_COSTLY_THROW && defined(TICK_USE_CBLAS))
// x and y are two pointers
#define CHECK_BLAS_OPTIMIZATION_PP(x, y, func_name)                               \
  if (typeid(*x) == typeid(*y) &&                                                 \
      (typeid(*x) == typeid(double) || typeid(*x) == typeid(float))) /* NOLINT */ \
    TICK_ERROR("" << func_name << " should use blas optimized version");

// x is a pointer, y is a scalar
#define CHECK_BLAS_OPTIMIZATION_PS(x, y, func_name)                               \
  if (typeid(*x) == typeid(y) &&                                                  \
      (typeid(*x) == typeid(double) || typeid(*x) == typeid(float))) /* NOLINT */ \
    TICK_ERROR("" << func_name << " should use blas optimized version");
#else
#define CHECK_BLAS_OPTIMIZATION_PP(x, y, func_name)
#define CHECK_BLAS_OPTIMIZATION_PS(x, y, func_name)
#endif

namespace tick {
namespace detail {

template <typename T>
template <typename K>
typename std::enable_if<std::is_same<T, std::atomic<K>>::value, T>::type
vector_operations_unoptimized<T>::dot(const ulong n, const T *x, const K *y) const {
  T result{0};
  for (uint64_t i = 0; i < n; ++i) {
    result += x[i].load() * y[i];
  }
  return result;
}
template <typename T>
template <typename K>
typename std::enable_if<std::is_same<T, std::atomic<K>>::value, K>::type
vector_operations_unoptimized<T>::dot(const ulong n, const K *x, const T *y) const {
  K result{0};
  for (uint64_t i = 0; i < n; ++i) {
    result += x[i] * y[i].load();
  }
  return result;
}

template <typename T>
template <typename K>
typename std::enable_if<!std::is_same<T, std::atomic<K>>::value, T>::type
vector_operations_unoptimized<T>::dot(const ulong n, const T *x, const K *y) const {
  CHECK_BLAS_OPTIMIZATION_PP(x, y, "dot prod");
  T result{0};
  for (uint64_t i = 0; i < n; ++i) {
    result += x[i] * y[i];
  }
  return result;
}

template <typename T>
template <typename K, typename Y>
typename std::enable_if<std::is_same<T, std::atomic<K>>::value &&
                        !std::is_same<Y, std::atomic<K>>::value>::type
vector_operations_unoptimized<T>::mult_incr(const uint64_t n, const K alpha, const Y *x,
                                            T *y) const {
  for (uint64_t i = 0; i < n; ++i) {
    K y_i = y[i].load();
    y_i += alpha * x[i];
    y[i].store(y_i);
  }
}

template <typename T>
template <typename K, typename Y>
typename std::enable_if<std::is_same<Y, std::atomic<K>>::value && !std::is_same<T, std::atomic<K>>::value>::type
vector_operations_unoptimized<T>::mult_incr(const uint64_t n, const K alpha, const Y *x, T *y) const {
  for (uint64_t i = 0; i < n; ++i) {
    K y_i = y[i];
    y_i += alpha * x[i].load();
    y[i] = y_i;
  }
}
template <typename T>
template <typename K, typename Y>
typename std::enable_if<std::is_same<T, std::atomic<K>>::value && std::is_same<Y, std::atomic<K>>::value>::type
vector_operations_unoptimized<T>::mult_incr(const uint64_t n, const K alpha, const Y *x, T *y) const {
  for (uint64_t i = 0; i < n; ++i) {
    K y_i = y[i].load();
    y_i += alpha * x[i].load();
    y[i].store(y_i);
  }
}
template <typename T>
template <typename K, typename Y>
typename std::enable_if<std::is_same<T, K>::value && std::is_same<Y, K>::value>::type
vector_operations_unoptimized<T>::mult_incr(const uint64_t n, const K alpha, const Y *x, T *y) const {
  CHECK_BLAS_OPTIMIZATION_PP(x, y, "mult_incr");
  for (uint64_t i = 0; i < n; ++i) {
    K y_i = y[i];
    y_i += alpha * x[i];
    y[i] = y_i;
  }
}

template <typename T>
template <typename K>
typename std::enable_if<std::is_same<T, std::atomic<K>>::value>::type
vector_operations_unoptimized<T>::set(const ulong n, const K alpha, T *x) const {
  for (uint64_t i = 0; i < n; ++i) x[i].store(alpha);
}
template <typename T>
template <typename K>
typename std::enable_if<!std::is_same<T, std::atomic<K>>::value>::type
vector_operations_unoptimized<T>::set(const ulong n, const K alpha, T *x) const {
#if (DEBUG_COSTLY_THROW && defined(TICK_USE_CBLAS) && defined(TICK_USE_CATLAS_))
  CHECK_BLAS_OPTIMIZATION_PS(x, alpha, "set");
#endif
  for (uint64_t i = 0; i < n; ++i) x[i] = alpha;
}

template <typename T>
template <typename K>
tick::promote_t<K> vector_operations_unoptimized<T>::sum(const ulong n, const T *x) const {
  return std::accumulate(x, x + n, tick::promote_t<K>{0});
}

template <typename T>
template <typename K>
typename std::enable_if<std::is_same<T, std::atomic<K>>::value>::type
vector_operations_unoptimized<T>::scale(const ulong n, const K alpha, T *x) const {
  for (uint64_t i = 0; i < n; ++i) {
    K x_i = x[i].load();
    x_i *= alpha;
    x[i].store(x_i);
  }
}
template <typename T>
template <typename K>
typename std::enable_if<!std::is_same<T, std::atomic<K>>::value>::type
vector_operations_unoptimized<T>::scale(const ulong n, const K alpha, T *x) const {
  CHECK_BLAS_OPTIMIZATION_PS(x, alpha, "scale");
  for (uint64_t i = 0; i < n; ++i) x[i] *= alpha;
}

template <typename T>
template <typename K>
typename std::enable_if<std::is_same<T, std::atomic<K>>::value>::type
vector_operations_unoptimized<T>::dot_matrix_vector_incr(const ulong m, const ulong n,
                                                         const K alpha, const T *a, const T *x,
                                                         const T beta, T *y) const {
  for (ulong i = 0; i < m; ++i) {
    K y_i = beta * y[i];
    for (ulong j = 0; j < n; ++j) {
      y_i += alpha * a[i * n + j] * x[j].load();
    }
    y[i].store(y_i);
  }
}

template <typename T>
template <typename K>
typename std::enable_if<!std::is_same<T, std::atomic<K>>::value>::type
vector_operations_unoptimized<T>::dot_matrix_vector_incr(const ulong m, const ulong n,
                                                         const K alpha, const T *a, const T *x,
                                                         const T beta, T *y) const {
  for (ulong i = 0; i < m; ++i) {
    y[i] = beta * y[i];
    for (ulong j = 0; j < n; ++j) {
      y[i] += alpha * a[i * n + j] * x[j];
    }
  }
}

template <typename T>
template <typename K>
typename std::enable_if<std::is_same<T, std::atomic<K>>::value>::type
vector_operations_unoptimized<T>::dot_matrix_vector(const ulong m, const ulong n, const K alpha,
                                                    const T *a, const T *x, T *y) const {
  for (ulong i = 0; i < m; ++i) {
    K y_i = 0;
    for (ulong j = 0; j < n; ++j) {
      y_i += alpha * a[i * n + j] * x[j].load();
    }
    y[i].store(y_i);
  }
}

template <typename T>
template <typename K>
typename std::enable_if<!std::is_same<T, std::atomic<K>>::value>::type
vector_operations_unoptimized<T>::dot_matrix_vector(const ulong m, const ulong n, const K alpha,
                                                    const T *a, const T *x, T *y) const {
  for (ulong i = 0; i < m; ++i) {
    y[i] = 0;
    for (ulong j = 0; j < n; ++j) {
      y[i] += alpha * a[i * n + j] * x[j];
    }
  }
}

#undef CHECK_BLAS_OPTIMIZATION_PP
#undef CHECK_BLAS_OPTIMIZATION_PS

}  //  end namespace detail
}  //  end namespace tick

#endif  // LIB_INCLUDE_TICK_ARRAY_VECTOR_OPS_UNOPTIMIZED_IMPL_H_
