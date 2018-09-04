

#ifndef LIB_INCLUDE_TICK_ARRAY_VECTOR_CBLAS_H_
#define LIB_INCLUDE_TICK_ARRAY_VECTOR_CBLAS_H_

#if defined(TICK_CBLAS_AVAILABLE)

namespace tick { namespace detail {

template <typename T>
struct vector_operations_cblas : vector_operations_unoptimized<T> {};

template <typename T>
struct vector_operations_cblas_base {
  template <typename K>
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

  template <typename T, typename K = T>
  typename std::enable_if<std::is_same<T, std::atomic<K>>::value, T>::type
  dot(const ulong n, const T *x, const K *y) const {
    return vector_operations_unoptimized<float>{}.dot(n, x, y);
  }

  template <typename T, typename K = T>
  typename std::enable_if<std::is_same<T, std::atomic<K>>::value, K>::type
  dot(const ulong n, const K *x, const T *y) const {
    return vector_operations_unoptimized<T>{}.dot(n, x, y);
  }

  template <typename T, typename K = T>
  typename std::enable_if<!std::is_same<T, std::atomic<K>>::value, T>::type
  dot(const ulong n, const T *x, const K *y) const {
    return cblas_sdot(n, x, 1, y, 1);
  }

  void scale(const ulong n, const std::atomic<float> alpha, float *x) const {
    cblas_sscal(n, alpha.load(), x, 1);
  }
  void scale(const ulong n, float alpha, float *x) const {
    cblas_sscal(n, alpha, x, 1);
  }

  template <typename T, typename K, typename Y>
  typename std::enable_if<std::is_same<T, std::atomic<K>>::value>::type
  mult_incr(const uint64_t n, const K alpha, const Y *x, T *y) const {
    return vector_operations_unoptimized<float>{}.mult_incr(n, alpha, x, y);
  }

  template <typename T, typename K, typename Y>
  typename std::enable_if<std::is_same<T, std::atomic<K>>::value && !std::is_same<Y, std::atomic<K>>::value>::type
  mult_incr(const uint64_t n, const K alpha, const Y *x, T *y) const {
    return vector_operations_unoptimized<K>{}.mult_incr(n, alpha, x, y);
  }

  template <typename T, typename K, typename Y>
  typename std::enable_if<std::is_same<Y, std::atomic<K>>::value && !std::is_same<T, std::atomic<K>>::value>::type
  mult_incr(const uint64_t n, const K alpha, const Y *x, T *y) const {
    return vector_operations_unoptimized<K>{}.mult_incr(n, alpha, x, y);
  }
  template <typename T, typename K, typename Y>
  typename std::enable_if<std::is_same<T, std::atomic<K>>::value && std::is_same<Y, std::atomic<K>>::value>::type
  mult_incr(const uint64_t n, const K alpha, const Y *x, T *y) const {
    return vector_operations_unoptimized<std::atomic<K>>{}.mult_incr(n, alpha, x, y);
  }

  template <typename T, typename K, typename Y>
  typename std::enable_if<!std::is_same<T, std::atomic<K>>::value && !std::is_same<Y, std::atomic<K>>::value>::type
  mult_incr(const uint64_t n, const K alpha, const Y *x, T *y) const {
    cblas_saxpy(n, alpha, x, 1, y, 1);
  }

#if defined(TICK_CATLAS_AVAILABLE)
  void set(const ulong n, const float alpha, float x) const override {
    catlas_sset(n, alpha, x, 1);
  }
#endif

  template <typename T, typename K = T>
  void dot_matrix_vector(const ulong m, const ulong n, const K alpha,
                                                      const T *a, const T *x, T *y) const {
    return vector_operations_unoptimized<K>{}.dot_matrix_vector(m, n, alpha, a, x, y);
  }

  template <typename T, typename K = T>
  void batch_multi_incr(size_t n_threads, const ulong b, const ulong n, const K *alpha, T **const x,
                        T *y) const {
    vector_operations_unoptimized<K>{}.batch_multi_incr(n_threads, b, n, alpha, x, y);
  }

  template <typename T, typename K = T>
  typename std::enable_if<std::is_same<T, std::atomic<K>>::value>::type dot_matrix_vector_incr(
      const ulong m, const ulong n, const T alpha, const T *a, const T *x, const T beta,
      T *y) const {
    return vector_operations_unoptimized<K>{}.dot_matrix_vector_incr(m, n, alpha, a, x, beta, y);
  }

  template <typename T, typename K = T>
  typename std::enable_if<!std::is_same<T, std::atomic<K>>::value>::type dot_matrix_vector_incr(
      const ulong m, const ulong n, const T alpha, const T *a, const T *x, const T beta,
      T *y) const {
    cblas_sgemv(CBLAS_ORDER::CblasRowMajor, CBLAS_TRANSPOSE::CblasNoTrans, m, n, alpha, a, n, x, 1,
                beta, y, 1);
  }
};

template <>
struct vector_operations_cblas<double> final
    : public vector_operations_cblas_base<double> {

  double absolute_sum(const ulong n, const double *x) const {
    return cblas_dasum(n, x, 1);
  }

  template <typename T, typename K = T>
  typename std::enable_if<std::is_same<T, std::atomic<K>>::value, T>::type
  dot(const ulong n, const T *x, const K *y) const {
    return vector_operations_unoptimized<K>{}.dot(n, x, y);
  }

  template <typename T, typename K = T>
  typename std::enable_if<std::is_same<T, std::atomic<K>>::value, K>::type
  dot(const ulong n, const K *x, const T *y) const {
    return vector_operations_unoptimized<T>{}.dot(n, x, y);
  }

  template <typename T, typename K = T>
  typename std::enable_if<!std::is_same<T, std::atomic<K>>::value, T>::type
  dot(const ulong n, const T *x, const K *y) const {
    return cblas_ddot(n, x, 1, y, 1);
  }

  void scale(const ulong n, const std::atomic<double> alpha, double *x) const {
    cblas_dscal(n, alpha.load(), x, 1);
  }
  void scale(const ulong n, double alpha, double *x) const {
    cblas_dscal(n, alpha, x, 1);
  }

  template <typename T, typename Y>
  typename std::enable_if<std::is_same<Y, std::atomic<double>>::value && !std::is_same<T, std::atomic<double>>::value>::type
  mult_incr(const uint64_t n, const double alpha, const Y *x, T *y) const {
    return vector_operations_unoptimized<double>{}.mult_incr(n, alpha, x, y);
  }

  template <typename T, typename Y>
  typename std::enable_if<std::is_same<T, std::atomic<double>>::value && !std::is_same<Y, std::atomic<double>>::value>::type
  mult_incr(const uint64_t n, const double alpha, const Y *x, T *y) const {
    return vector_operations_unoptimized<double>{}.mult_incr(n, alpha, x, y);
  }

  template <typename T, typename Y>
  typename std::enable_if<std::is_same<T, std::atomic<double>>::value && std::is_same<Y, std::atomic<double>>::value>::type
  mult_incr(const uint64_t n, const double alpha, const Y *x, T *y) const {
    return vector_operations_unoptimized<std::atomic<double>>{}.mult_incr(n, alpha, x, y);
  }

  template <typename T, typename Y>
  typename std::enable_if<!std::is_same<T, std::atomic<double>>::value && !std::is_same<Y, std::atomic<double>>::value>::type
  mult_incr(const uint64_t n, const double alpha, const Y *x, T *y) const {
    cblas_daxpy(n, alpha, x, 1, y, 1);
  }


#if defined(TICK_CATLAS_AVAILABLE)
  void set(const ulong n, const T alpha, double *x) const override { catlas_sset(n, alpha, x, 1); }
#endif

  template <typename T, typename K = T>
  void dot_matrix_vector(const ulong m, const ulong n, const K alpha,
                                                      const T *a, const T *x, T *y) const {
    return vector_operations_unoptimized<K>{}.dot_matrix_vector(m, n, alpha, a, x, y);
  }

  template <typename T, typename K = T>
  void batch_multi_incr(size_t n_threads, const ulong b, const ulong n, const K *alpha, T **const x,
                        T *y) const {
    vector_operations_unoptimized<K>{}.batch_multi_incr(n_threads, b, n, alpha, x, y);
  }

  template <typename T, typename K = T>
  typename std::enable_if<std::is_same<T, std::atomic<K>>::value>::type dot_matrix_vector_incr(
      const ulong m, const ulong n, const T alpha, const T *a, const T *x, const T beta,
      T *y) const {
    vector_operations_unoptimized<K>{}.dot_matrix_vector_incr(m, n, alpha, a, x, beta, y);
  }

  template <typename T, typename K = T>
  typename std::enable_if<!std::is_same<T, std::atomic<K>>::value>::type dot_matrix_vector_incr(
      const ulong m, const ulong n, const T alpha, const T *a, const T *x, const T beta,
      T *y) const {
    cblas_dgemv(CBLAS_ORDER::CblasRowMajor, CBLAS_TRANSPOSE::CblasNoTrans, m, n, alpha, a, n, x, 1,
                beta, y, 1);
  }
};

}  // namespace detail

template <typename T>
using vector_operations = detail::vector_operations_cblas<T>;

}  // namespace tick

#endif  // defined(TICK_CBLAS_AVAILABLE)

#endif  // LIB_INCLUDE_TICK_ARRAY_VECTOR_CBLAS_H_

