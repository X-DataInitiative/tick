#ifndef TICK_BASE_ARRAY_SRC_CBLAS_WRAPPERS_H_
#define TICK_BASE_ARRAY_SRC_CBLAS_WRAPPERS_H_

#include <numeric>

#include "defs.h"
#include "promote.h"

#ifdef __APPLE__

#  include <Accelerate/Accelerate.h>
#  define XDATA_CBLAS_AVAILABLE

// TODO(svp) Disabling this feature until we find a good way to determine if ATLAS is actually available
//#  define XDATA_CATLAS_AVAILABLE

#elif defined(HAVE_CBLAS)

#  ifdef __has_include
#    if __has_include ("cblas.h")
#      include <cblas.h>
#      define XDATA_CBLAS_AVAILABLE
#    endif
#  endif

#endif

template<typename T>
struct cblas_wrappers_base {
    virtual T dot(const ulong n, const T *x, const T *y) const {
        T result{0};

        for (ulong i = 0; i < n; ++i) {
            result += x[i] * y[i];
        }

        return result;
    }

    virtual tick::promote_t<T> sum(const ulong n, const T *x) const {
        return std::accumulate(x, x + n, tick::promote_t<T>{0});
    }

    virtual void scale(const ulong n, const T alpha, T *x) const {
        for (ulong i = 0; i < n; ++i) {
            x[i] *= alpha;
        }
    }

    virtual void set(const ulong n, const T alpha, T *x) const {
        for (ulong i = 0; i < n; ++i) {
            x[i] = alpha;
        }
    }
};

template<typename T>
struct cblas_wrappers : public cblas_wrappers_base<T> {};

#if defined(XDATA_CBLAS_AVAILABLE)
template<>
struct cblas_wrappers<float> final : public cblas_wrappers_base<float> {
    float absolute_sum(const ulong n, const float *x) const {
        return cblas_sasum(n, x, 1);
    }

    float dot(const ulong n, const float *x, const float *y) const override {
        return cblas_sdot(n, x, 1, y, 1);
    }

    void scale(const ulong n, const float alpha, float *x) const override {
        cblas_sscal(n, alpha, x, 1);
    }

#if defined(XDATA_CATLAS_AVAILABLE)
    void set(const ulong n, const float alpha, float* x) const override {
        catlas_sset(n, alpha, x, 1);
    }
#endif
};

template<>
struct cblas_wrappers<double> final : public cblas_wrappers_base<double> {
    double absolute_sum(const ulong n, const double *x) const {
        return cblas_dasum(n, x, 1);
    }

    double dot(const ulong n, const double *x, const double *y) const override {
        return cblas_ddot(n, x, 1, y, 1);
    }

    void scale(const ulong n, const double alpha, double *x) const override {
        cblas_dscal(n, alpha, x, 1);
    }

#if defined(XDATA_CATLAS_AVAILABLE)
    void set(const ulong n, const double alpha, double* x) const override {
        catlas_dset(n, alpha, x, 1);
    }
#endif
};

#endif  // defined(XDATA_CBLAS_AVAILABLE)

#endif  // TICK_BASE_ARRAY_SRC_CBLAS_WRAPPERS_H_
