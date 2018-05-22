

#include "tick/array/array.h"
#include "tick/array/sparsearray.h"



// @brief Returns the scalar product of the array with `array`
template <typename T>
template <typename K>
T BaseArray<T>::dot(const BaseArray<K> &array) const {
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

template <>
template <>
double BaseArray<double>::dot<std::atomic<double>>(
    const BaseArray<std::atomic<double>> &array) const {
  if (_size != array.size()) TICK_ERROR("Arrays don't have the same size");

  double result = 0;

  // Case dense/dense
  if (is_dense() && array.is_dense()) {
    return (tick::vector_operations<double>{})
        .template dot<std::atomic<double>>(this->size(), this->data(),
                                           array.data());
  }

  // Case sparse/sparse
  if (is_sparse() && array.is_sparse()) {
    auto _size_gt_array_size_lambda =
        [&](const SparseArray<double> *a1,
            const SparseArray<std::atomic<double>> *a2) {
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
        [&](const SparseArray<std::atomic<double>> *a1,
            const SparseArray<double> *a2) {
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
          static_cast<const SparseArray<double> *>(this),
          static_cast<const SparseArray<std::atomic<double>> *>(&array));
    } else {
      _size_lt_array_size_lambda(
          static_cast<const SparseArray<std::atomic<double>> *>(&array),
          static_cast<const SparseArray<double> *>(this));
    }

    return result;
  }

  // Case sparse/dense
  auto is_dense_lambda = [&](const SparseArray<std::atomic<double>> *sa,
                             const Array<double> *da) {
    for (uint64_t i = 0; i < sa->size_sparse(); i++) {
      result += sa->data()[i].load() * da->data()[sa->indices()[i]];
    }
  };
  auto is_sparse_lambda = [&](const Array<double> *sa,
                              const SparseArray<std::atomic<double>> *da) {
    for (uint64_t i = 0; i < sa->size_sparse(); i++) {
      result += sa->data()[i] * da->data()[sa->indices()[i]].load();
    }
  };
  if (is_dense()) {
    is_dense_lambda(
        static_cast<const SparseArray<std::atomic<double>> *>(&array),
        static_cast<const Array<double> *>(this));
  } else {
    is_sparse_lambda(
        static_cast<const Array<double> *>(this),
        static_cast<const SparseArray<std::atomic<double>> *>(&array));
  }
  return result;
}

template <>
template <>
float BaseArray<float>::dot<std::atomic<float>>(
    const BaseArray<std::atomic<float>> &array) const {
  if (_size != array.size()) TICK_ERROR("Arrays don't have the same size");

  float result = 0;

  // Case dense/dense
  if (is_dense() && array.is_dense()) {
    return (tick::vector_operations<float>{})
        .template dot<std::atomic<float>>(this->size(), this->data(),
                                          array.data());
  }

  // Case sparse/sparse
  if (is_sparse() && array.is_sparse()) {
    auto _size_gt_array_size_lambda =
        [&](const SparseArray<float> *a1,
            const SparseArray<std::atomic<float>> *a2) {
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
        [&](const SparseArray<std::atomic<float>> *a1,
            const SparseArray<float> *a2) {
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
          static_cast<const SparseArray<float> *>(this),
          static_cast<const SparseArray<std::atomic<float>> *>(&array));
    } else {
      _size_lt_array_size_lambda(
          static_cast<const SparseArray<std::atomic<float>> *>(&array),
          static_cast<const SparseArray<float> *>(this));
    }

    return result;
  }

  // Case sparse/dense
  auto is_dense_lambda = [&](const SparseArray<float> *sa,
                             const Array<std::atomic<float>> *da) {
    for (uint64_t i = 0; i < sa->size_sparse(); i++) {
      result += sa->data()[i] * da->data()[sa->indices()[i]].load();
    }
  };
  auto is_sparse_lambda = [&](const Array<std::atomic<float>> *sa,
                              const SparseArray<float> *da) {
    for (uint64_t i = 0; i < sa->size_sparse(); i++) {
      result += sa->data()[i].load() * da->data()[sa->indices()[i]];
    }
  };
  if (is_dense()) {
    is_dense_lambda(static_cast<const SparseArray<float> *>(this),
                    static_cast<const Array<std::atomic<float>> *>(&array));
  } else {
    is_sparse_lambda(static_cast<const Array<std::atomic<float>> *>(&array),
                     static_cast<const SparseArray<float> *>(this));
  }
  return result;
}

template DLL_PUBLIC int BaseArray<int>::dot<int>(BaseArray<int> const &) const;

template DLL_PUBLIC int64_t
BaseArray<int64_t>::dot<int64_t>(BaseArray<int64_t> const &) const;

template DLL_PUBLIC short BaseArray<short>::dot<short>(
    BaseArray<short> const &) const;

template DLL_PUBLIC float BaseArray<float>::dot<float>(
    BaseArray<float> const &) const;

template DLL_PUBLIC double BaseArray<double>::dot<double>(
    BaseArray<double> const &) const;

template DLL_PUBLIC unsigned int BaseArray<unsigned int>::dot<unsigned int>(
    BaseArray<unsigned int> const &) const;

template DLL_PUBLIC uint64_t
BaseArray<uint64_t>::dot<uint64_t>(BaseArray<uint64_t> const &) const;

template DLL_PUBLIC unsigned short BaseArray<unsigned short>::dot<
    unsigned short>(BaseArray<unsigned short> const &) const;

#if defined(_MSC_VER)
template DLL_PUBLIC double BaseArray<double>::dot<std::atomic<double> >(
    BaseArray<std::atomic<double> > const &) const;
template DLL_PUBLIC float BaseArray<float>::dot<std::atomic<float> >(
    BaseArray<std::atomic<float> > const &) const;
#endif
