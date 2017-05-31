//
//  dot.h
//  TICK
//

#ifndef TICK_BASE_ARRAY_SRC_DOT_H_
#define TICK_BASE_ARRAY_SRC_DOT_H_

#include "array.h"
#include "sparsearray.h"
#include "vector_operations.h"

// @brief Returns the scalar product of the array with `array`
template<typename T>
T BaseArray<T>::dot(const BaseArray<T> &array) const {
    if (_size != array.size()) TICK_ERROR("Arrays don't have the same size");

    T result = 0;

    // Case dense/dense
    if (is_dense() && array.is_dense()) {
        return (tick::vector_operations<T>{}).dot(this->size(), this->data(), array.data());
    }

    // Case sparse/sparse
    if (is_sparse() && array.is_sparse()) {
        const SparseArray<T> *a1;
        const SparseArray<T> *a2;

        // put the more sparse in a1
        if (_size > array.size()) {
            a1 = static_cast<const SparseArray<T> *>(&array);
            a2 = static_cast<const SparseArray<T> *>(this);
        } else {
            a2 = static_cast<const SparseArray<T> *>(&array);
            a1 = static_cast<const SparseArray<T> *>(this);
        }

        ulong i1 = 0, i2 = 0;

        while (true) {
            if (i1 >= a1->size_sparse()) break;
            while (i2 < a2->size_sparse() && a2->indices()[i2] < a1->indices()[i1]) i2++;
            if (i2 >= a2->size_sparse()) break;
            if (a2->indices()[i2] == a1->indices()[i1]) {
                result += a2->data()[i2] * a1->data()[i1];
                i1++;
            } else {
                while (i1 < a1->size_sparse() && a2->indices()[i2] > a1->indices()[i1]) i1++;
            }
        }

        return result;
    }

    // Case sparse/dense
    const SparseArray<T> *sa;
    const Array<T> *da;

    if (is_dense()) {
        sa = static_cast<const SparseArray<T> *>(&array);
        da = static_cast<const Array<T> *>(this);
    } else {
        sa = static_cast<const SparseArray<T> *>(this);
        da = static_cast<const Array<T> *>(&array);
    }
    for (ulong i = 0; i < sa->size_sparse(); i++) {
        result += sa->data()[i] * da->data()[sa->indices()[i]];
    }
    return result;
}

#endif  // TICK_BASE_ARRAY_SRC_DOT_H_
