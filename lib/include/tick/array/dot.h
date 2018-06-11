//
//  dot.h
//  TICK
//

#ifndef LIB_INCLUDE_TICK_ARRAY_DOT_H_
#define LIB_INCLUDE_TICK_ARRAY_DOT_H_

#include "tick/array/array.h"
#include "tick/array/sparsearray.h"

// @brief Returns the scalar product of the array with `array`
template <typename T>
template <typename Y>
typename std::enable_if<!std::is_same<T, bool>::value && !std::is_same<Y, bool>::value && !std::is_same<T, std::atomic<Y>>::value, Y>::type
BaseArray<T>::dot(const BaseArray<K> &array) const {
  if (_size != array.size()) TICK_ERROR("Arrays don't have the same size");

  T result = 0;

  // Case dense/dense
  if (is_dense() && array.is_dense()) {
    return (tick::vector_operations<T>{})
        .dot(this->size(), this->data(), array.data());
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

    uint64_t i1 = 0, i2 = 0;

    while (true) {
      if (i1 >= a1->size_sparse()) break;
      while (i2 < a2->size_sparse() && a2->indices()[i2] < a1->indices()[i1])
        i2++;
      if (i2 >= a2->size_sparse()) break;
      if (a2->indices()[i2] == a1->indices()[i1]) {
        result += a2->data()[i2] * a1->data()[i1];
        i1++;
      } else {
        while (i1 < a1->size_sparse() && a2->indices()[i2] > a1->indices()[i1])
          i1++;
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
  for (uint64_t i = 0; i < sa->size_sparse(); i++) {
    result += sa->data()[i] * da->data()[sa->indices()[i]];
  }
  return result;
}

template <typename T>
template <typename Y>
typename std::enable_if<std::is_same<Y, std::atomic<T>>::value, T>::type
BaseArray<T>::dot(const BaseArray<Y> &array) const {
  if (_size != array.size()) TICK_ERROR("Arrays don't have the same size");

  T result = 0;

  // Case dense/dense
  if (is_dense() && array.is_dense()) {
    return (tick::vector_operations<T>{})
        .template dot<std::atomic<T>>(this->size(), this->data(),
                                           array.data());
  }

  // Case sparse/sparse
  if (is_sparse() && array.is_sparse()) {
    auto _size_gt_array_size_lambda =
        [&](const SparseArray<T> *a1,
            const SparseArray<std::atomic<T>> *a2) {
          uint64_t i1 = 0, i2 = 0;
          while (true) {
            if (i1 >= a1->size_sparse()) break;
            while (i2 < a2->size_sparse() &&
                   a2->indices()[i2] < a1->indices()[i1])
              i2++;
            if (i2 >= a2->size_sparse()) break;
            if (a2->indices()[i2] == a1->indices()[i1]) {
              result += a2->data()[i2].load() * a1->data()[i1];
              i1++;
            } else {
              while (i1 < a1->size_sparse() &&
                     a2->indices()[i2] > a1->indices()[i1])
                i1++;
            }
          }
        };

    auto _size_lt_array_size_lambda =
        [&](const SparseArray<std::atomic<T>> *a1,
            const SparseArray<T> *a2) {
          uint64_t i1 = 0, i2 = 0;
          while (true) {
            if (i1 >= a1->size_sparse()) break;
            while (i2 < a2->size_sparse() &&
                   a2->indices()[i2] < a1->indices()[i1])
              i2++;
            if (i2 >= a2->size_sparse()) break;
            if (a2->indices()[i2] == a1->indices()[i1]) {
              result += a2->data()[i2] * a1->data()[i1].load();
              i1++;
            } else {
              while (i1 < a1->size_sparse() &&
                     a2->indices()[i2] > a1->indices()[i1])
                i1++;
            }
          }
        };

    // put the more sparse in a1
    if (_size > array.size()) {
      _size_gt_array_size_lambda(
          static_cast<const SparseArray<T> *>(this),
          static_cast<const SparseArray<std::atomic<T>> *>(&array));
    } else {
      _size_lt_array_size_lambda(
          static_cast<const SparseArray<std::atomic<T>> *>(&array),
          static_cast<const SparseArray<T> *>(this));
    }

    return result;
  }

  // Case sparse/dense
  auto is_dense_lambda = [&](const SparseArray<std::atomic<T>> *sa,
                             const Array<T> *da) {
    for (uint64_t i = 0; i < sa->size_sparse(); i++) {
      result += sa->data()[i].load() * da->data()[sa->indices()[i]];
    }
  };
  auto is_sparse_lambda = [&](const Array<T> *sa,
                              const SparseArray<std::atomic<T>> *da) {
    for (uint64_t i = 0; i < sa->size_sparse(); i++) {
      result += sa->data()[i] * da->data()[sa->indices()[i]].load();
    }
  };
  if (is_dense()) {
    is_dense_lambda(
        static_cast<const SparseArray<std::atomic<T>> *>(&array),
        static_cast<const Array<T> *>(this));
  } else {
    is_sparse_lambda(
        static_cast<const Array<T> *>(this),
        static_cast<const SparseArray<std::atomic<T>> *>(&array));
  }
  return result;
}
#endif  // LIB_INCLUDE_TICK_ARRAY_DOT_H_
