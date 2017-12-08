// License: BSD 3 clause


#ifdef PYTHON_LINK
#include <Python.h>
#else
#define Py_BEGIN_ALLOW_THREADS
#define Py_END_ALLOW_THREADS
#endif

#include "tick/array/array.h"

// The non parametric estimation is based on the following quantities :
//
// Given two point processes Y and Z, computes (for all i) the Signal
//
//      E((Y[t+lags[i+1]+epsilon]-Y[t+lags[i]+epsilon]) | zmin<=Z[t]<=zmax) - Lambda_Y
//
// where epsilon is infinitely small
//
//  - lags : array of lags to loop on. It is of size N+1, so it represents N intervals.
//  - y_T : end time of Y signal
//  - y_lambda : unconditionnal intensity of Y
//
// The result is stored in res_X (abscissa) and res_Y (ordinates) : they are of size N
//
// Returns the number of terms involved in computation.
//

// No mark for y !!

void PointProcessCondLaw(ArrayDouble &y_time,
                         ArrayDouble &z_time, ArrayDouble &z_mark,
                         ArrayDouble &lags,
                         double zmin, double zmax,
                         double y_T,
                         double y_lambda,
                         ArrayDouble &res_X, ArrayDouble &res_Y) {
  if (z_time.size() != z_mark.size()) {
    TICK_ERROR("z_time (size=" << z_time.size() << ") and z_mark (size=" << z_mark.size()
                               << ") should have the same size");
  }

  if (res_X.size() != res_Y.size()) {
    TICK_ERROR("res_X (size=" << res_X.size() << ") and res_Y (size=" << res_Y.size()
                              << ") should have the same size");
  }

  if (res_X.size() + 1 != lags.size()) {
    TICK_ERROR(
      "lags (size=" << lags.size() << ") should be of the size of res_X (size=" << res_X.size()
                    << " plus one");
  }

  ulong y_index_lag;
  ulong y_index_lag_delta;
  double ytlag;
  double ytlagdelta;
  double lag;
  ulong N = lags.size() - 1;

  ArrayULong tab_y_index = ArrayULong(N);

  ulong n_terms = 0;

  Py_BEGIN_ALLOW_THREADS

  // Inititialization of the resulting array
  res_Y.fill(0);

  double lagMax = lags[N];

  // To improve performance by remembering the last point of Y in each time slice
  tab_y_index.fill(0);

  // The loop on the jumps of Z
  ulong y_index = 0;
  for (ulong z_index = 0; z_index < z_mark.size(); z_index++) {
    // Is it an eligible jump ?
    if (zmin < zmax && z_index > 0 &&
      (zmin > z_mark[z_index] - z_mark[z_index - 1]
        || z_mark[z_index] - z_mark[z_index - 1] > zmax))
      continue;

    // Brings y_index after z_t
    // After this block, one has
    // time(Y[y_index]) >= z_t
    double z_t = z_time[z_index];
    if (z_t + lagMax >= y_T) break;
    n_terms += 1;
    while (y_index < y_time.size() && y_time[y_index] < z_t) y_index++;
    if (y_index >= y_time.size()) break;

    y_index_lag_delta = y_index;

    // Loop on the lag
    for (ulong k = 0; k < N; k++) {
      lag = lags[k];
      y_index_lag = y_index_lag_delta;

      while (y_time[y_index_lag] <= z_t + lag) {
        y_index_lag++;
        if (y_index_lag >= y_time.size()) break;
      }
      if (y_index_lag >= y_time.size()) {
        y_index_lag_delta = y_time.size() - 1;
        tab_y_index[k] = y_index_lag_delta;
        continue;
      }
      ytlag = (y_index_lag == 0 ? 0 : y_index_lag - 1);

      y_index_lag_delta = std::max(y_index_lag, tab_y_index[k]);
      while (y_time[y_index_lag_delta] <= z_t + lags[k + 1]) {
        y_index_lag_delta++;
        if (y_index_lag_delta >= y_time.size()) break;
      }
      if (y_index_lag_delta >= y_time.size()) {
        y_index_lag_delta = y_time.size() - 1;
        if (y_index_lag == y_time.size() - 1) {
          tab_y_index[k] = y_index_lag_delta;
          continue;
        }
      }
      tab_y_index[k] = y_index_lag_delta;
      ytlagdelta = (y_index_lag_delta == 0 ? 0 : y_index_lag_delta - 1);
      res_Y[k] += ytlagdelta - ytlag;
    }
  }

  for (ulong k = 0; k < N; k++) {
    if (n_terms != 0) {
      res_Y[k] /= n_terms;
    }
    res_Y[k] /= (lags[k + 1] - lags[k]);
    res_Y[k] -= y_lambda;
    res_X[k] = (lags[k + 1] + lags[k]) / 2.0;
  }

  Py_END_ALLOW_THREADS
}

//
// Given two point processes Y and Z, computes the Signal
//
//      E((Y[t+delta]-Y[t+epsilon]) | zmin<=Z[t]<=zmax) - Lambda_Y*delta
//
// where epsilon is infinitely small
//
double PointProcessCondLawSingle(ArrayDouble &y_time,
                                 ArrayDouble &y_mark,
                                 ArrayDouble &z_time,
                                 ArrayDouble &z_mark,
                                 double delta,
                                 double zmin,
                                 double zmax) {
  double res = 0;
  Py_BEGIN_ALLOW_THREADS

  const double lambda = y_time.size() / (y_time[y_time.size() - 1] - y_time[0]);

  // The loop on the jumps of Z
  std::int64_t count = 0;
  ulong y_index = 0;
  ulong y_index_delta = 0;
  for (ulong z_index = 0; z_index < z_time.size(); z_index++) {
    // Is it an eligible jump ?
    if (zmin < zmax && z_index > 0 &&
      (zmin > z_mark[z_index] - z_mark[z_index - 1]
        || z_mark[z_index] - z_mark[z_index - 1] > zmax))
      continue;

    double z_t = z_time[z_index];

    // Looking for index of y[t]
    while (y_index < y_mark.size() && y_time[y_index] < z_t) y_index++;

    if (y_index >= y_time.size())
      break;

    double yt;

    if (y_mark[y_index] == z_t)
      yt = y_mark[y_index];
    else
      yt = (y_index == 0 ? 0 : y_mark[y_index - 1]);

    if (z_t + delta >= y_mark[y_mark.size() - 1]) break;

    count += 1;

    // Looking for index for y[t+delta]
    y_index_delta = std::max(y_index_delta, y_index);
    while (y_mark[y_index_delta] <= z_t + delta) y_index_delta++;

    double ytdelta = 0;

    if (y_mark[y_index_delta] == z_t + delta)
      ytdelta = y_mark[y_index_delta];
    else
      ytdelta = (y_index_delta == 0 ? 0 : y_mark[y_index_delta - 1]);

    res += ytdelta - yt - lambda * delta;
  }

  res /= count;

  Py_END_ALLOW_THREADS

  return res;
}

