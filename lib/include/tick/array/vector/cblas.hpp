

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

#if (!defined(__APPLE__))
  void solve_linear_system(int n, T *A, T *b, int* ipiv=nullptr) const {
    return vector_operations_unoptimized<T>{}.template solve_linear_system(n, A, b, ipiv);
  }

  void solve_symmetric_linear_system(int n, T *A, T *b, int* ipiv=nullptr) const {
    return vector_operations_unoptimized<T>{}.template solve_linear_system(n, A, b);
  }
#endif
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
  typename std::enable_if<std::is_same<T,std::atomic<K>>::value, K>::type
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

#if (defined(__APPLE__))
  void solve_linear_system(int n, float *A, float *b, int* ipiv=nullptr) const {
    int n_cols = 1;
    int lda = n;
    bool should_free = false;
    if (ipiv == nullptr) {
      ipiv = new int[n];
      should_free = true;
    }
    int ldb = n;
    int info;
    sgesv_(&n, &n_cols, A, &lda, ipiv, b, &ldb, &info);
    if (should_free) delete[] ipiv;

    if (info != 0) {
      TICK_ERROR("Linear solver failed with info=" << info)
    }
  }

  void solve_symmetric_linear_system(int n, float *A, float *b, int* ipiv=nullptr) const {
    // it seems faster this way with BLAS
    if (n < 30) return solve_linear_system(n, A, b, ipiv);
    int n_cols = 1;
    int lda = n;
    int ldb = n;
    int info;
    char UPLO = 'L';
    sposv_(&UPLO, &n, &n_cols, A, &lda, b, &ldb, &info);
    if (info != 0) {
      TICK_ERROR("Symmetric linear solver failed with info=" << info)
    }
  }

#endif

#if defined(TICK_CATLAS_AVAILABLE)
  void set(const ulong n, const float alpha, float x) const override {
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

  template <typename T, typename K = T>
  typename std::enable_if<std::is_same<T, std::atomic<K>>::value, T>::type
  dot(const ulong n, const T *x, const K *y) const {
    return vector_operations_unoptimized<K>{}.dot(n, x, y);
  }

  template <typename T, typename K = T>
  typename std::enable_if<std::is_same<T,std::atomic<K>>::value, K>::type
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


#if (defined(__APPLE__))
  void solve_linear_system(int n, double *A, double *b, int* ipiv=nullptr) const {
    int n_cols = 1;
    int lda = n;
    bool should_free = false;
    if (ipiv == nullptr) {
      ipiv = new int[n];
      should_free = true;
    }
    int ldb = n;
    int info;
    dgesv_(&n, &n_cols, A, &lda, ipiv, b, &ldb, &info);
    if (info != 0) {
      TICK_ERROR("Linear solver failed with info=" << info)
    }

    if (should_free) delete[] ipiv;
  }

  void solve_symmetric_linear_system(int n, double *A, double *b, int* ipiv=nullptr) const {
    // it seems faster this way with BLAS
    if (n < 30) return solve_linear_system(n, A, b, ipiv);
    int n_cols = 1;
    int lda = n;
    int ldb = n;
    int info;
    char UPLO = 'L';
    dposv_(&UPLO, &n, &n_cols, A, &lda, b, &ldb, &info);
    if (info != 0) {
      TICK_ERROR("Symmetric linear solver failed with info=" << info)
    }
  }
#endif


#if defined(TICK_CATLAS_AVAILABLE)
  void set(const ulong n, const T alpha, double *x) const override {
    catlas_sset(n, alpha, x, 1);
  }
#endif
};

}  // namespace detail

template <typename T>
using vector_operations = detail::vector_operations_cblas<T>;

}  // namespace tick

#endif  // defined(TICK_CBLAS_AVAILABLE)
