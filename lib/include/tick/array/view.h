//
//  view.h
//  tests
//
//  Created by bacry on 03/01/2016.
//  Copyright (c) 2016 bacry. All rights reserved.
//

/// @file

#ifndef LIB_INCLUDE_TICK_ARRAY_VIEW_H_
#define LIB_INCLUDE_TICK_ARRAY_VIEW_H_

// License: BSD 3 clause

#include "array.h"
#include "sparsearray.h"
#include "sarray.h"

/**
 * \defgroup view_mod View management
 * \brief List of all the functions for creating views
 * @{
 */

//
// Array
//

//! @brief Returns a full view on an Array
//! \warning A view does not own its allocations
template<typename T>
Array<T> view(Array<T> &a) {
    return Array<T>(a.size(), a.data());
}

//! @brief Returns a partial view on an Array
//! \param a The array the view is built on
//! \param first The first index of the view
//! \warning by definition a view does not own its allocations
template<typename T>
Array<T> view(Array<T> &a, ulong first) {
    if (first >= a.size()) TICK_BAD_INDEX(0, a.size(), first);
    return Array<T>(a.size() - first, a.data() + first);
}

//! @brief Returns a partial view on an Array
//! \param a The array the view is built on
//! \param first The fist index of the view
//! \param last The last index (excluded) of the view
//! \warning by definition a view does not own its allocations
template<typename T>
Array<T> view(const Array<T> &a, ulong first, ulong last) {
    if (first >= a.size()) TICK_BAD_INDEX(0, a.size(), first);
    if (last > a.size()) TICK_BAD_INDEX(0, a.size(), last);
    if (first >= last) return Array<T>();
    return Array<T>(last - first, a.data() + first);
}

//
// SparseArray
//

//! @brief Returns a full view on an SparseArray
//! \warning A view does not own its allocations
template<typename T>
SparseArray<T> view(SparseArray<T> &a) {
    return SparseArray<T>(a.size(), a.size_sparse(), a.indices(), a.data());
}

//
// BaseArray
//

//! @brief Returns a full view on an BaseArray
//! \warning A view does not own its allocations
template<typename T>
BaseArray<T> view(BaseArray<T> &a) {
    if (a.is_dense())
        return view(static_cast<Array<T> &>(a));
    else
        return view(static_cast<SparseArray<T> &>(a));
}

//! @brief Returns a partial view on an BaseArray
//! \param a The array the view is built on
//! \param first The fist index of the view
//! \param first The last index of the view
//! \warning by definition a view does not own its allocations
template<typename T>
BaseArray<T> view(BaseArray<T> &a, ulong first, ulong last) {
    if (a.is_sparse())
        TICK_ERROR("Cannot make a partial view of a SparseArray");

    return view(static_cast<Array<T> &>(a), first, last);
}

//! @}

#endif  // LIB_INCLUDE_TICK_ARRAY_VIEW_H_
