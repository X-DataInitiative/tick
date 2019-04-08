
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
void solve_linear_system_with_gauss_jordan(int n, T *A, T *b) {
  // Unsure behavior if T is not double or float
  // From https://martin-thoma.com/solving-linear-equations-with-gaussian-elimination/
  // Switched from row major to column major and extract b from last column of A
  for (int i = 0; i < n; i++) {
    // Search for maximum in this row
    T max_element = std::abs(A[i * n + i]);
    int max_row = i;
    for (int k = i + 1; k < n; k++) {
      if (abs(A[i * n + k]) > max_element) {
        max_element = abs(A[i * n + k]);
        max_row = k;
      }
    }

    // Swap maximum row with current row (column by column)
    for (int k = i; k < n; k++) {
      T tmp = A[k * n + max_row];
      A[k * n + max_row] = A[k * n + i];
      A[k * n + i] = tmp;
    }
    T tmp = b[max_row];
    b[max_row] = b[i];
    b[i] = tmp;

    // Make all rows below this one 0 in current column
    for (int k = i + 1; k < n; k++) {
      double c = -A[i * n + k] / A[i * n + i];
      for (int j = i; j < n + 1; j++) {
        if (i == j) {
          A[j * n + k] = 0;
        }
        if (j < n) {
          A[j * n + k] += c * A[j * n + i];
        } else {
          b[k] += c * b[i];
        }
      }
    }
  }

  // Solve equation Ax=b for an upper triangular matrix A
  for (int i = n - 1; i >= 0; i--) {
    b[i] /= A[i * n + i];
    for (int k = i - 1; k >= 0; k--) {
      b[k] -= A[i * n + k] * b[i];
    }
  }
}

template <typename T>
void vector_operations_unoptimized<T>::solve_linear_system(int n, T *A, T *b, int *ipiv) const {
  solve_linear_system_with_gauss_jordan(n, A, b);
}

template <typename T>
void vector_operations_unoptimized<T>::solve_positive_symmetric_linear_system(
    int n, T *A, T *b, int *ipiv, int switch_linear) const {

  solve_linear_system_with_gauss_jordan(n, A, b);
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
