// License: BSD 3 clause

//
// Created by Maryan Morel on 15/05/2017.
//

#include <mutex>
#include "tick/preprocessing/longitudinal_features_lagger_mp.h"

LongitudinalFeaturesLagger_MP::LongitudinalFeaturesLagger_MP(ulong n_intervals,
                                                             SArrayULongPtr _n_lags, size_t n_jobs)
    : LongitudinalPreprocessor(n_jobs),
      n_intervals(n_intervals),
      n_lags(_n_lags),
      n_features(_n_lags->size()),
      n_lagged_features(_n_lags->size() + _n_lags->sum()) {
  if (n_lags != nullptr) compute_col_offset(n_lags);
  n_output_features = get_n_output_features();
}

void LongitudinalFeaturesLagger_MP::compute_col_offset(const SArrayULongPtr n_lags) {
  ArrayULong col_offset_temp = ArrayULong(n_lags->size());
  col_offset_temp.init_to_zero();
  for (ulong i(1); i < n_lags->size(); i++) {
    if ((*n_lags)[i] > n_intervals) {  // (*n_lags)[i] >= n_intervals
      TICK_ERROR("n_lags elements must be between 0 and n_intervals.");  // (n_intervals - 1) was
                                                                         // actually wrong?
    }
    col_offset_temp[i] = col_offset_temp[i - 1] + (*n_lags)[i - 1] + 1;
  }
  col_offset = col_offset_temp.as_sarray_ptr();
}

void LongitudinalFeaturesLagger_MP::dense_lag_preprocessor(ArrayDouble2d &features,
                                                           ArrayDouble2d &out,
                                                           ulong censoring) const {
  if (n_intervals != features.n_rows()) {
    TICK_ERROR("Features matrix rows count should match n_intervals.");
  }
  if (n_features != features.n_cols()) {
    TICK_ERROR("Features matrix column count should match n_lags length.");
  }
  if (out.n_cols() != n_lagged_features) {
    TICK_ERROR("n_columns of &out should be equal to n_features + sum(n_lags).");
  }
  if (out.n_rows() != n_intervals) {
    TICK_ERROR("n_rows of &out is inconsistent with n_intervals");
  }
  ulong n_cols_feature, row, col, max_col;
  double value;
  for (ulong feature = 0; feature < n_features; feature++) {
    n_cols_feature = (*n_lags)[feature] + 1;
    for (ulong j = 0; j < n_intervals; j++) {
      row = j;
      col = (*col_offset)[feature];
      // use view_row instead of (row, feature) to be const
      value = view_row(features, row)[feature];
      max_col = col + n_cols_feature;
      if (value != 0) {
        while (row < censoring && col < max_col) {
          out[row * n_lagged_features + col] = value;
          row++;
          col++;
        }
      }
    }
  }
}

void LongitudinalFeaturesLagger_MP::sparse_lag_preprocessor(ArrayULong &row, ArrayULong &col,
                                                            ArrayDouble &data, ArrayULong &out_row,
                                                            ArrayULong &out_col,
                                                            ArrayDouble &out_data,
                                                            ulong censoring) const {
  // TODO: add checks here ? Or do them in Python ?
  if (row.size() != col.size() || col.size() != data.size() || data.size() != row.size())
    TICK_ERROR("row, col and data arrays should have the same size (coo matrix)");
  if (out_row.size() != out_col.size() || out_col.size() != out_data.size() ||
      out_data.size() != out_row.size())
    TICK_ERROR("out_row, out_col and out_data arrays should have the same size (coo matrix)");

  ulong j(0), r, c, offset, new_col, max_col;
  double value;

  for (ulong i = 0; i < data.size(); i++) {
    value = data[i];
    r = row[i];
    c = col[i];
    offset = (*col_offset)[c];
    max_col = offset + (*n_lags)[c] + 1;
    new_col = offset;

    while (r < censoring && new_col < max_col) {
      out_row[j] = r;
      out_col[j] = new_col;
      out_data[j] = value;
      r++;
      new_col++;
      j++;
    }
  }
}

ulong LongitudinalFeaturesLagger_MP::get_n_output_features() {
  ulong arraysum = 0;
  std::vector<ulong> arraysumint;
  for (size_t i = 0; i < n_lags->size(); i++) arraysumint.push_back((*n_lags)[i] + 1);
  for (ulong i : arraysumint) arraysum += i;
  return arraysum;
}

SSparseArrayDouble2dPtr LongitudinalFeaturesLagger_MP::sparse_lagger(
    SSparseArrayDouble2dPtr &feature_matrix, ulong censoring_i) {
  if (censoring_i > n_intervals || censoring_i < 1)
    TICK_ERROR("censoring shoud be an integer in [1, n_intervals]");

  CooMatrix<double> coo(feature_matrix);

  // TODO FIX this is wrong, but the coo.toSparse removes all the useless zero data.
  ulong estimated_nnz = coo.nnz * n_output_features;

  // std::cout << "estimated nnz : " << estimated_nnz << " nnz : " << coo.nnz
  //           << " arraysum : " << n_output_features << std::endl;

  ArrayULong out_row(estimated_nnz);
  ArrayULong out_col(estimated_nnz);
  ArrayDouble out_data(estimated_nnz);

  out_row.init_to_zero();
  out_col.init_to_zero();
  out_data.init_to_zero();

  sparse_lag_preprocessor(coo.rows, coo.cols, coo.data, out_row, out_col, out_data, censoring_i);

  coo.rows = out_row;
  coo.cols = out_col;
  coo.data = out_data;

  return coo.toSparse(n_intervals, n_output_features);
}

void LongitudinalFeaturesLagger_MP::transform_thread_dense(
    std::vector<ArrayDouble2d> splited_features, std::vector<ArrayDouble2d> &output,
    std::mutex &thread_mutex, std::vector<ulong> splited_censoring) {
  for (ulong i = 0; i < splited_features.size(); i++) {
    ArrayDouble2d transformed(splited_features[i].n_rows(), n_output_features);
    transformed.init_to_zero();
    dense_lag_preprocessor(splited_features[i], transformed, splited_censoring[i]);
    thread_mutex.lock();  // just in case, needed ?
    output.push_back(transformed);
    thread_mutex.unlock();
  }
}

void LongitudinalFeaturesLagger_MP::transform_thread_sparse(
    std::vector<SSparseArrayDouble2dPtr> splited_features,
    std::vector<SSparseArrayDouble2dPtr> &output, std::mutex &thread_mutex,
    std::vector<ulong> splited_censoring) {
  for (ulong i = 0; i < splited_features.size(); i++) {
    SSparseArrayDouble2dPtr transformed = sparse_lagger(splited_features[i], splited_censoring[i]);
    thread_mutex.lock();  // just in case, needed ?
    output.push_back(transformed);
    thread_mutex.unlock();
  }
}

std::vector<ArrayDouble2d> LongitudinalFeaturesLagger_MP::transform(
    std::vector<ArrayDouble2d> features, std::vector<ulong> censoring) {
  if (features.empty()) TICK_ERROR("features is empty");

  if (censoring.empty()) {
    for (ulong i = 0; i < features.size(); i++) censoring.push_back(n_intervals);
  }

  if (features.size() != censoring.size())
    TICK_ERROR("features size and censoring size doesn\'t match");

  std::pair<ulong, ulong> base_shape = {features[0].n_rows(), features[0].n_cols()};
  for (ArrayDouble2d f : features)
    if (f.n_rows() != base_shape.first || f.n_cols() != base_shape.second)
      TICK_ERROR("All the elements of features should have the same shape");

  size_t thread_count = std::min((size_t)features.size(), n_jobs);
  std::vector<std::vector<ArrayDouble2d>> splited_features = split_vector(features, thread_count);
  features.clear();
  std::vector<std::vector<ulong>> splited_censoring = split_vector(censoring, thread_count);
  censoring.clear();

  if (splited_features.size() != splited_censoring.size())
    TICK_ERROR("Unexepected error : splited_features.size() != splited_censoring.size()");
  if (splited_features.size() != thread_count || splited_censoring.size() != thread_count)
    TICK_ERROR(
        "Unexepected error : splited_features.size() != thread_count || splited_censoring.size() "
        "!= thread_count");
  if (splited_features.empty() || splited_censoring.empty())
    TICK_ERROR("Unexepected error : splited_features.empty() || splited_censoring.empty()");

  std::vector<ArrayDouble2d> output;
  std::vector<std::thread> threads;
  std::mutex thread_mutex;

  for (size_t i = 0; i < thread_count; i++)
    threads.push_back(std::thread(&LongitudinalFeaturesLagger_MP::transform_thread_dense, this,
                                  splited_features[i], std::ref(output), std::ref(thread_mutex),
                                  splited_censoring[i]));

  splited_features.clear();
  splited_censoring.clear();

  for (size_t i = 0; i < threads.size(); i++) threads[i].join();

  return output;
}

std::vector<SSparseArrayDouble2dPtr> LongitudinalFeaturesLagger_MP::transform(
    std::vector<SSparseArrayDouble2dPtr> features, std::vector<ulong> censoring) {
  if (features.empty()) TICK_ERROR("features is empty");

  if (censoring.empty()) {
    for (ulong i = 0; i < features.size(); i++) censoring.push_back(n_intervals);
  }

  if (features.size() != censoring.size())
    TICK_ERROR("features size and censoring size doesn\'t match");

  std::pair<ulong, ulong> base_shape = {features[0]->n_rows(), features[0]->n_cols()};
  n_intervals = base_shape.first;
  for (SSparseArrayDouble2dPtr f : features)
    if (f->n_rows() != base_shape.first || f->n_cols() != base_shape.second)
      TICK_ERROR("All the elements of features should have the same shape");

  size_t thread_count = std::min((size_t)features.size(), n_jobs);
  std::vector<std::vector<SSparseArrayDouble2dPtr>> splited_features =
      split_vector(features, thread_count);
  features.clear();
  std::vector<std::vector<ulong>> splited_censoring = split_vector(censoring, thread_count);
  censoring.clear();

  if (splited_features.size() != splited_censoring.size())
    TICK_ERROR("Unexepected error : splited_features.size() != splited_censoring.size()");
  if (splited_features.empty() || splited_censoring.empty())
    TICK_ERROR("Unexepected error : splited_features.empty() || splited_censoring.empty()");

  std::vector<SSparseArrayDouble2dPtr> output;
  std::vector<std::thread> threads;
  std::mutex thread_mutex;

  for (size_t i = 0; i < thread_count; i++)
    threads.push_back(std::thread(&LongitudinalFeaturesLagger_MP::transform_thread_sparse, this,
                                  splited_features[i], std::ref(output), std::ref(thread_mutex),
                                  splited_censoring[i]));

  splited_features.clear();
  splited_censoring.clear();

  for (size_t i = 0; i < threads.size(); i++) threads[i].join();

  return output;
}
