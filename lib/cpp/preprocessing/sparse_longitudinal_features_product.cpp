// License: BSD 3 clause

//
// Created by Maryan Morel on 11/05/2017.
//

#include "tick/preprocessing/sparse_longitudinal_features_product.h"
#include <map>

SparseLongitudinalFeaturesProduct::SparseLongitudinalFeaturesProduct(
    const SBaseArrayDouble2dPtrList1D &features)
    : n_features(features[0]->n_cols()) {}

ulong SparseLongitudinalFeaturesProduct::get_feature_product_col(ulong col1,
                                                                 ulong col2,
                                                                 ulong n_cols) const {
  if (col1 > col2) {  // ensure we have the right order as the following formula is not symmetric
    col1 += col2;
    col2 = col1 - col2;
    col1 -= col2;
  }
  if (col1 == col2) {
    TICK_ERROR("col1 index == col2 index in feature product")
  }
  ulong offset = static_cast<ulong>((col1 + 1) *
      (n_cols - static_cast<double>(col1) / 2.0));
  return offset + (col2 - col1 - 1);
}

void SparseLongitudinalFeaturesProduct::sparse_features_product(ArrayULong &row,
                                                                 ArrayULong &col,
                                                                 ArrayDouble &data,
                                                                 ArrayULong &out_row,
                                                                 ArrayULong &out_col,
                                                                 ArrayDouble &out_data) const {
  // Create a map[col]: max nnz row idx
  std::map<ulong, ulong> col_last_nnz;
  std::vector<ulong> keys;
  ulong r, c, k(0);
  double d = 1;  // data always equal to 1 with this hypothesis
  for (ulong i=0; i < col.size(); i++) {
    r = row[i];
    c = col[i];
    out_row[k] = r;
    out_col[k] = c;
    out_data[k] = d;
    k++;

    if (col_last_nnz.find(c) == col_last_nnz.end()) {
      // not found
      col_last_nnz[c] = 0;
      keys.push_back(c);
    }

    // update the map
    if (col_last_nnz[c] < r) {
      col_last_nnz[c] = r;
    }
  }

  // get keys of map
  ulong c1, c2;
  // for i, c1 in enum(keys):
  ulong n_keys = keys.size();
  for (ulong i=0; i < (n_keys - 1); i++) {
    // for c2 in enum(keys[i:]:
    for (ulong j=(i+1); j < n_keys; j++) {
      // idx = get_comp_idx(c1, c2)
      c1 = keys[i];
      c2 = keys[j];
      c = get_feature_product_col(c1, c2, n_features);
      // row = max(map(c1), map(c2))
      r = col_last_nnz[c1] > col_last_nnz[c2] ? col_last_nnz[c1] : col_last_nnz[c2];
      // insert 1 in out[row, idx]
      out_row[k] = r;
      out_col[k] = c;
      out_data[k] = d;
      k++;
    }
  }
}
