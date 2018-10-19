//
//  view2d.h
//  tests
//
//  Created by bacry on 07/01/2016.
//  Copyright (c) 2016 bacry. All rights reserved.
//

#ifndef LIB_INCLUDE_TICK_ARRAY_VIEW2D_H_
#define LIB_INCLUDE_TICK_ARRAY_VIEW2D_H_

// License: BSD 3 clause

#include "array2d.h"

/**
 * \defgroup view2_mod View2d management
 * \brief List of all the functions for creating views on 2d Arrays
 * @{
 */

//
// Array2d
//

//! @brief Returns a full view on an Array2d
//! \warning A view does not own its allocations
template <typename T, typename MAJ>
Array2d<T, MAJ> view(Array2d<T, MAJ> &a) {
  return Array2d<T, MAJ>(a.n_rows(), a.n_cols(), a.data());
}

//! @brief Returns a 1d Array view of a row of a row major Array2d
//! \warning A view does not own its allocations
template <typename T, typename MAJ>
typename std::enable_if<std::is_same<MAJ, RowMajor>::value, Array<T>>::type
view_row(Array2d<T, MAJ> &a, ulong i) {
  if (i >= a.n_rows()) TICK_BAD_INDEX(0, i, a.n_rows());
  return Array<T>(a.n_cols(), a.data() + i * a.n_cols());
}

//! @brief Returns a 1d Array view of a column of a col major Array2d
//! \warning A view does not own its allocations
template <typename T, typename MAJ>
typename std::enable_if<std::is_same<MAJ, ColMajor>::value, Array<T>>::type
view_col(Array2d<T, MAJ> &a, ulong i) {
  if (i >= a.n_cols()) TICK_BAD_INDEX(0, i, a.n_cols());
  return Array<T>(a.n_rows(), a.data() + i * a.n_rows());
}

//
// SparseArray2d
//

//! @brief Returns a full view on an SparseArray2d
//! \warning A view does not own its allocations
template <typename T, typename MAJ>
SparseArray2d<T, MAJ> view(SparseArray2d<T, MAJ> &a) {
  return SparseArray2d<T, MAJ>(a.n_rows(), a.n_cols(), a.row_indices(), a.indices(),
                          a.data());
}

//! @brief Returns a view on the ith row of a SparseArray2d with row major
//! \warning A view does not own its allocations
template <typename T, typename MAJ>
typename std::enable_if<std::is_same<MAJ, RowMajor>::value, SparseArray<T>>::type
view_row(SparseArray2d<T, MAJ> &a, ulong i) {
  if (a.row_indices()[i + 1] - a.row_indices()[i] == 0)
    return SparseArray<T>(a.n_cols(), 0, nullptr, nullptr);

  return SparseArray<T>(a.n_cols(), a.row_indices()[i + 1] - a.row_indices()[i],
                        a.indices() + a.row_indices()[i],
                        a.data() + a.row_indices()[i]);
}

//! @brief Returns a view on the ith row of a SparseArray2d with column major
//! \warning A view does not own its allocations
template <typename T, typename MAJ>
typename std::enable_if<std::is_same<MAJ, ColMajor>::value, SparseArray<T>>::type
view_col(SparseArray2d<T, MAJ> &a, ulong i) {
  if (a.row_indices()[i + 1] - a.row_indices()[i] == 0)
    return SparseArray<T>(a.n_rows(), 0, nullptr, nullptr);

  return SparseArray<T>(a.n_rows(), a.row_indices()[i + 1] - a.row_indices()[i],
                        a.indices() + a.row_indices()[i],
                        a.data() + a.row_indices()[i]);
}


//
// BaseArray2d
//

//! @brief Returns a full view on an BaseArray2d
//! \warning A view does not own its allocations
template <typename T, typename MAJ>
BaseArray2d<T, MAJ> view(BaseArray2d<T, MAJ> &a) {
  if (a.is_dense())
    return view(static_cast<Array2d<T, MAJ> &>(a));
  else
    return view(static_cast<SparseArray2d<T, MAJ> &>(a));
}

//! @brief Returns a 1d BaseArray view of a row of an BaseArray2d
//! \warning A view does not own its allocations
template <typename T, typename MAJ>
typename std::enable_if<std::is_same<MAJ, RowMajor>::value, BaseArray<T>>::type
view_row(BaseArray2d<T, MAJ> &a, ulong i) {
  if (a.is_dense())
    return view_row(static_cast<Array2d<T, MAJ> &>(a), i);
  else
    return view_row(static_cast<SparseArray2d<T, MAJ> &>(a), i);
}

template <typename T>
Array2d<T> view_rows(Array2d<T> &a, ulong start, ulong end) {
  if (start >= a.n_rows()) TICK_BAD_INDEX(0, start, a.n_rows());
  if (end < start || end >= a.n_rows()) TICK_BAD_INDEX(start, end, a.n_rows());
  const ulong n_rows = end - start;
  return Array2d<T>(n_rows, a.n_cols(), a.data() + start * a.n_cols());
}

template <typename T>
SparseArray2d<T> view_rows(SparseArray2d<T> &a, ulong start, ulong end) {
  if (start >= a.n_rows()) TICK_BAD_INDEX(0, start, a.n_rows());
  if (end < start || end >= a.n_rows()) TICK_BAD_INDEX(start, end, a.n_rows());
  const ulong n_rows = end - start;

  if (a.row_indices()[end] - a.row_indices()[start] == 0)
    return SparseArray2d<T>(n_rows, a.n_cols(), 0, nullptr, nullptr);

  TICK_ERROR(
      "Row indices should be shifted by n_rows. Would be cool to do it with no copy by "
      "adding a new attribute: row_shift");
  return SparseArray2d<T>(n_rows, a.n_cols(), a.row_indices(), a.indices() + a.row_indices()[start],
                          a.data() + a.row_indices()[start]);
}

template <typename T>
BaseArray2d<T> view_rows(BaseArray2d<T> &a, ulong start, ulong end) {
  if (a.is_dense())
    return view_rows(static_cast<Array2d<T> &>(a), start, end);
  else
    return view_rows(static_cast<SparseArray2d<T> &>(a), start, end);
}

//! @}

#endif  // LIB_INCLUDE_TICK_ARRAY_VIEW2D_H_
