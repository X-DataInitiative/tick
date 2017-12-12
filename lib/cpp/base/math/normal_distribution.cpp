// License: BSD 3 clause

//
// Created by Martin Bompaire on 11/04/16.
//

#include "tick/base/defs.h"

#include <cmath>
#include "tick/base/math/normal_distribution.h"

// Standard Gaussian cumulative density function, based on
// Handbook of Mathematical Functions
// Abramowitz and Stegun
// formula 7.1.26
double standard_normal_cdf(double x) {
  // constants
  static const double a1 = 0.254829592;
  static const double a2 = -0.284496736;
  static const double a3 = 1.421413741;
  static const double a4 = -1.453152027;
  static const double a5 = 1.061405429;
  static const double p = 0.3275911;
  static const double sqrt_2 = sqrt(2);

  int sign_x = 1;
  if (x < 0)
    sign_x = -1;
  x = fabs(x) / sqrt_2;

  double t = 1.0 / (1.0 + p * x);
  double y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * exp(-x * x);
  return 0.5 * (1.0 + sign_x * y);
}

/* Beasley-Springer Moro algorithm for computing the inverse
 * cumulative normal function */
double standard_normal_inv_cdf(const double x) {
  if (x > 0 && x < 1) {
    // initialization
    static const double a[4] = {2.50662823884, -18.61500062529, 41.39119773534,
                                -25.44106049637};
    static const double b[4] = {-8.47351093090, 23.08336743743, -21.06224101826,
                                3.13082909833};
    static const double c[9] = {0.3374754822726147, 0.9761690190917186, 0.1607979714918209,
                                0.0276438810333863, 0.0038405729373609, 0.0003951896411919,
                                0.0000321767881768, 0.0000002888167364, 0.0000003960315187};
    double y = x - 0.5;
    double r, s, t, result;
    if (fabs(y) < 0.42) {
      // Beasley-Springer
      r = y * y;
      result = y * (a[0] + r * (a[1] + r * (a[2] + r * a[3])))
          / (1.0 + r * (b[0] + r * (b[1] + r * (b[2] + r * b[3]))));
    } else {
      // Moro
      if (y <= 0) {
        r = x;
      } else {
        r = 1 - x;
      }
      s = log(-log(r));
      t = c[0]
          + s * (c[1] + s * (c[2] + s * (c[3] + s * (c[4] + s * (c[5] + s * (c[6] + s * (c[7] + s * c[8])))))));
      if (x > 0.5) {
        result = t;
      } else {
        result = -t;
      }
    }
    return result;
  } else {
    TICK_ERROR("Inverse CDF cannot be computed at 0 or 1.");
  }
}

void standard_normal_inv_cdf(ArrayDouble &q, ArrayDouble &out) {
  for (ulong i = 0; i < q.size(); i++) {
    out[i] = standard_normal_inv_cdf(q[i]);
  }
}
