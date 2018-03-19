// License: BSD 3 clause

#include "tick/prox/prox_tv.h"

// This piece comes from L. Condat's paper, see tick's documentation
template <class T>
void TProxTV<T>::call(const Array<T> &coeffs, T step, Array<T> &out,
                      ulong start, ulong end) {
  Array<T> sub_coeffs = view(coeffs, start, end);
  Array<T> sub_out = view(out, start, end);

  const T width = sub_coeffs.size();
  const T thresh = step * strength;

  if (width > 0) { /*to avoid invalid memory access to input[0]*/
    int k = 0,
        k0 = 0; /*k: current sample location, k0: beginning of current segment*/
    T umin = thresh, umax = -thresh; /*u is the dual variable*/
    T vmin = sub_coeffs[0] - thresh,
      vmax = sub_coeffs[0] + thresh; /*bounds for the segment's value*/
    int kplus = 0,
        kminus =
            0; /*last positions where umax=-lambda, umin=lambda, respectively*/
    const T twolambda = 2.0 * thresh; /*auxiliary variable*/
    const T minlambda = -thresh;      /*auxiliary variable*/
    for (;;) {                        /*simple loop, the exit test is inside*/
      while (k == width - 1) {        /*we use the right boundary condition*/
        if (umin < 0.0) { /*vmin is too high -> negative jump necessary*/
          do {
            sub_out[k0++] = vmin;
          } while (k0 <= kminus);
          umax = (vmin = sub_coeffs[kminus = k = k0]) + (umin = thresh) - vmax;
        } else if (umax > 0.0) { /*vmax is too low -> positive jump necessary*/
          do {
            sub_out[k0++] = vmax;
          } while (k0 <= kplus);
          umin =
              (vmax = sub_coeffs[kplus = k = k0]) + (umax = minlambda) - vmin;
        } else {
          vmin += umin / (k - k0 + 1);
          do {
            sub_out[k0++] = vmin;
          } while (k0 <= k);
          if (positive) {
            for (ulong i = start; i < end; i++) {
              if (out[i] < 0) {
                out[i] = 0;
              }
            }
          }
          return;
        }
      }
      if ((umin += sub_coeffs[k + 1] - vmin) <
          minlambda) { /*negative jump necessary*/
        do {
          sub_out[k0++] = vmin;
        } while (k0 <= kminus);

        vmax = (vmin = sub_coeffs[kplus = kminus = k = k0]) + twolambda;
        umin = thresh;
        umax = minlambda;
      } else if ((umax += sub_coeffs[k + 1] - vmax) >
                 thresh) { /*positive jump necessary*/
        do {
          sub_out[k0++] = vmax;
        } while (k0 <= kplus);
        vmin = (vmax = sub_coeffs[kplus = kminus = k = k0]) - twolambda;
        umin = thresh;
        umax = minlambda;
      } else { /*no jump necessary, we continue*/
        k++;
        if (umin >= thresh) { /*update of vmin*/
          vmin += (umin - thresh) / ((kminus = k) - k0 + 1);
          umin = thresh;
        }
        if (umax <= minlambda) { /*update of vmax*/
          vmax += (umax + thresh) / ((kplus = k) - k0 + 1);
          umax = minlambda;
        }
      }
    }
  }
}

template <class T>
T TProxTV<T>::value(const Array<T> &coeffs, ulong start, ulong end) {
  T diff, tv_norm = 0;
  for (ulong i = start + 1; i < end; i++) {
    diff = coeffs[i] - coeffs[i - 1];
    if (diff > 0) tv_norm += diff;
    if (diff < 0) tv_norm -= diff;
  }
  return strength * tv_norm;
}

template class DLL_PUBLIC TProxTV<double>;
template class DLL_PUBLIC TProxTV<float>;
