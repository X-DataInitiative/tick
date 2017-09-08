//
//  view2d.h
//  tests
//
//  Created by bacry on 07/01/2016.
//  Copyright (c) 2016 bacry. All rights reserved.
//

#ifndef TICK_BASE_ARRAY_SRC_VIEW2D_H_
#define TICK_BASE_ARRAY_SRC_VIEW2D_H_

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
template<typename T>
Array2d<T> view(Array2d<T> &a) {
    return Array2d<T>(a.n_rows(), a.n_cols(), a.data());
}

//! @brief Returns a full view on an Array2d
//! Const version
//! \warning A view does not own its allocations
template<typename T>
Array2d<T> view(const Array2d<T> &a) {
    return Array2d<T>(a.n_rows(), a.n_cols(), a.data());
}

//! @brief Returns a 1d Array view of a row of an Array2d
//! \warning A view does not own its allocations
template<typename T>
Array<T> view_row(Array2d<T> &a, ulong i) {
    if (i >= a.n_rows()) TICK_BAD_INDEX(0, i, a.n_rows());
    return Array<T>(a.n_cols(), a.data() + i * a.n_cols());
}

//! @brief Returns a 1d Array view of a row of an Array2d
//! Const version
//! \warning A view does not own its allocations
template<typename T>
Array<T> view_row(const Array2d<T> &a, const ulong i) {
    if (i >= a.n_rows()) TICK_BAD_INDEX(0, i, a.n_rows());
    return Array<T>(a.n_cols(), a.data() + i * a.n_cols());
}

//
// SparseArray2d
//

//! @brief Returns a full view on an SparseArray2d
//! \warning A view does not own its allocations
template<typename T>
SparseArray2d<T> view(SparseArray2d<T> &a) {
    return SparseArray2d<T>(a.n_rows(), a.n_cols(), a.row_indices(), a.indices(),
                            a.data());
}

//! @brief Returns a full view on an SparseArray2d
//! Const version
//! \warning A view does not own its allocations
template<typename T>
SparseArray2d<T> view(const SparseArray2d<T> &a) {
    return SparseArray2d<T>(a.n_rows(), a.n_cols(), a.row_indices(), a.indices(),
                            a.data());
}

//! @brief Returns a view on the ith row of a SparseArray2d
//! \warning A view does not own its allocations
template<typename T>
SparseArray<T> view_row(SparseArray2d<T> &a, ulong i) {
    if (a.row_indices()[i + 1] - a.row_indices()[i] == 0)
        return SparseArray<T>(a.n_cols(), 0, nullptr, nullptr);

    return SparseArray<T>(a.n_cols(),
                          a.row_indices()[i + 1] - a.row_indices()[i],
                          a.indices() + a.row_indices()[i],
                          a.data() + a.row_indices()[i]);
}

//! @brief Returns a view on the ith row of a SparseArray2d
//! Const version
//! \warning A view does not own its allocations
template<typename T>
SparseArray<T> view_row(const SparseArray2d<T> &a, const ulong i) {
    if (a.row_indices()[i + 1] - a.row_indices()[i] == 0)
        return SparseArray<T>(a.n_cols(), 0, nullptr, nullptr);

    return SparseArray<T>(a.n_cols(),
                          a.row_indices()[i + 1] - a.row_indices()[i],
                          a.indices() + a.row_indices()[i],
                          a.data() + a.row_indices()[i]);
}


//
// BaseArray2d
//

//! @brief Returns a full view on an BaseArray2d
//! \warning A view does not own its allocations
template<typename T>
BaseArray2d<T> view(BaseArray2d<T> &a) {
    if (a.is_dense())
        return view(static_cast<Array2d<T> &>(a));
    else
        return view(static_cast<SparseArray2d<T> &>(a));
}

//! @brief Returns a full view on an BaseArray2d
//! Const version
//! \warning A view does not own its allocations
template<typename T>
BaseArray2d<T> view(const BaseArray2d<T> &a) {
    if (a.is_dense())
        return view(static_cast<const Array2d<T> &>(a));
    else
        return view(static_cast<const SparseArray2d<T> &>(a));
}


//! @brief Returns a 1d BaseArray view of a row of an BaseArray2d
//! \warning A view does not own its allocations
template<typename T>
BaseArray<T> view_row(BaseArray2d<T> &a, ulong i) {
    if (a.is_dense())
        return view_row(static_cast<Array2d<T> &>(a), i);
    else
        return view_row(static_cast<SparseArray2d<T> &>(a), i);
}

//! @brief Returns a 1d BaseArray view of a row of an BaseArray2d
//! Const version
//! \warning A view does not own its allocations
template<typename T>
BaseArray<T> view_row(const BaseArray2d<T> &a, ulong i) {
    if (a.is_dense())
        return view_row(static_cast<const Array2d<T> &>(a), i);
    else
        return view_row(static_cast<const SparseArray2d<T> &>(a), i);
}

//! @}

#endif  // TICK_BASE_ARRAY_SRC_VIEW2D_H_
