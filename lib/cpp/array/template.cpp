

#include "tick/array/array.h"

template <typename T>
template <typename K>
void Array<T>::SET_DATA_INDEX(Array<T> &arr, size_t index, K value) {
  arr._data[index] = value;
}
template <>
template <>
void Array<std::atomic<double>>::SET_DATA_INDEX<double>(
    Array<std::atomic<double>> &arr, size_t index, double value) {
  arr._data[index].store(value);
}
template <>
template <>
void Array<std::atomic<float>>::SET_DATA_INDEX<float>(
    Array<std::atomic<float>> &arr, size_t index, float value) {
  arr._data[index].store(value);
}

template <typename T>
template <typename K>
void Array<T>::SET_T_FROM_K(T &t, K k) {
  t = k;
}
template <>
template <>
void Array<std::atomic<double>>::SET_T_FROM_K<double>(std::atomic<double> &t,
                                                      double k) {
  t.store(k);
}
template <>
template <>
void Array<std::atomic<float>>::SET_T_FROM_K<float>(std::atomic<float> &t,
                                                    float k) {
  t.store(k);
}

template <typename T>
void Array<T>::SET_T_FROM_T(T &t1, T &t2) {
  t1 = t2;
}
template <>
void Array<std::atomic<double>>::SET_T_FROM_T(std::atomic<double> &t1,
                                              std::atomic<double> &t2) {
  t1.store(t2.load());
}
template <>
void Array<std::atomic<float>>::SET_T_FROM_T(std::atomic<float> &t1,
                                             std::atomic<float> &t2) {
  t1.store(t2.load());
}

template <class T>
template <class K>
void Array<T>::set_data_index(size_t index, K value) {
  _data[index] = value;
}
template <>
template <>
void Array<std::atomic<float>>::set_data_index(size_t index, float value) {
  _data[index].store(value);
}
template <>
template <>
void Array<std::atomic<double>>::set_data_index(size_t index, double value) {
  _data[index].store(value);
}

template <class T>
template <class K>
void Array<T>::mult_incr(const BaseArray<T> &x, const K a) {
  if (this->size() != x.size()) {
    TICK_ERROR("Vectors don't have the same size.");
  } else {
    if (x.is_sparse()) {
      for (uint64_t j = 0; j < x.size_sparse(); j++) {
        _data[x.indices()[j]] += x.data()[j] * a;
      }
    } else {
      tick::vector_operations<T>{}.template mult_incr<K>(
          this->size(), a, x.data(), this->data());
    }
  }
}

template <>
template <>
void Array<std::atomic<double>>::mult_incr(
    const BaseArray<std::atomic<double>> &x, const double a) {
  if (this->size() != x.size()) {
    TICK_ERROR("Vectors don't have the same size.");
  } else {
    if (x.is_sparse()) {
      for (uint64_t j = 0; j < x.size_sparse(); j++) {
        double x_indices_j = _data[x.indices()[j]].load();
        x_indices_j += x.data()[j] * a;
        _data[x.indices()[j]].store(x_indices_j);
      }
    } else {
      tick::vector_operations<std::atomic<double>>{}.template mult_incr<double>(
          this->size(), a, x.data(), this->data());
    }
  }
}
template <>
template <>
void Array<std::atomic<float>>::mult_incr(
    const BaseArray<std::atomic<float>> &x, const float a) {
  if (this->size() != x.size()) {
    TICK_ERROR("Vectors don't have the same size.");
  } else {
    if (x.is_sparse()) {
      for (uint64_t j = 0; j < x.size_sparse(); j++) {
        float x_indices_j = _data[x.indices()[j]].load();
        x_indices_j += x.data()[j] * a;
        _data[x.indices()[j]].store(x_indices_j);
      }
    } else {
      tick::vector_operations<std::atomic<float>>{}.template mult_incr<float>(
          this->size(), a, x.data(), this->data());
    }
  }
}

// sort array inplace
template <typename T>
void Array<T>::sort(bool increasing) {
  if (increasing)
    std::sort(_data, _data + _size);
  else
    std::sort(_data, _data + _size, std::greater<T>());
}
template <>
void Array<std::atomic<double>>::sort(bool increasing) {
  TICK_CLASS_DOES_NOT_IMPLEMENT("Array<std::atomic<double>>");
}
template <>
void Array<std::atomic<float>>::sort(bool increasing) {
  TICK_CLASS_DOES_NOT_IMPLEMENT("Array<std::atomic<float>>");
}

template class Array<double>;
template class Array<float>;
template class Array<short>;
template class Array<uint64_t>;
template class Array<unsigned short>;

// sort array given a sort function
template <typename T>
template <typename F>
void Array<T>::sort_function(Array<uint64_t> &index, F order_function) {
  std::vector<value_index<T>> pairs(_size);
  for (uint64_t i = 0; i < _size; ++i) {
    SET_T_FROM_T(pairs[i].first, _data[i]);
    pairs[i].second = i;
  }

  std::sort(pairs.begin(), pairs.end(), order_function);

  for (uint64_t i = 0; i < _size; ++i) {
    SET_T_FROM_T(_data[i], pairs[i].first);
    index[i] = pairs[i].second;
  }
}

template <>
template <typename F>
void Array<std::atomic<double>>::sort_function(Array<uint64_t> &index,
                                               F order_function) {
  TICK_CLASS_DOES_NOT_IMPLEMENT("Array<std::atomic<double>>");
}
template <>
template <typename F>
void Array<std::atomic<float>>::sort_function(Array<uint64_t> &index,
                                              F order_function) {
  TICK_CLASS_DOES_NOT_IMPLEMENT("Array<std::atomic<float>>");
}

template class Array<std::atomic<double>>;
template class Array<std::atomic<float>>;

template DLL_PUBLIC void Array<int>::mult_incr(const BaseArray<int> &x,
                                               const int a);

template DLL_PUBLIC void Array<int64_t>::mult_incr(const BaseArray<int64_t> &x,
                                                   const int64_t a);

template DLL_PUBLIC void Array<float>::mult_incr(const BaseArray<float> &x,
                                                 const float a);

template DLL_PUBLIC void Array<float>::mult_incr<double>(
    const BaseArray<float> &x, const double a);

template DLL_PUBLIC void Array<float>::mult_incr<int>(const BaseArray<float> &x,
                                                      const int a);

template DLL_PUBLIC void Array<short>::mult_incr(const BaseArray<short> &x,
                                                 const short a);

template DLL_PUBLIC void Array<double>::mult_incr(const BaseArray<double> &x,
                                                  const double a);

template DLL_PUBLIC void Array<double>::mult_incr(const BaseArray<double> &x,
                                                  int);

template DLL_PUBLIC void Array<unsigned int>::mult_incr(
    const BaseArray<unsigned int> &x, const unsigned int a);

template DLL_PUBLIC void Array<uint64_t>::mult_incr(
    const BaseArray<uint64_t> &x, const uint64_t a);

template DLL_PUBLIC void Array<unsigned short>::mult_incr(
    const BaseArray<unsigned short> &x, const unsigned short a);

#if defined(_MSC_VER)

template DLL_PUBLIC void Array<std::atomic<float>>::mult_incr<float>(
    BaseArray<std::atomic<float>> const &, float);

template DLL_PUBLIC void Array<std::atomic<double>>::mult_incr<double>(
    BaseArray<std::atomic<double>> const &, double);

#endif

template DLL_PUBLIC void Array<int>::sort(bool);

template DLL_PUBLIC void Array<int64_t>::sort(bool);

template DLL_PUBLIC void Array<unsigned int>::sort(bool);

template DLL_PUBLIC void Array<float>::sort_function(
    Array<uint64_t> &, bool (*)(std::pair<float, uint64_t> const &,
                                std::pair<float, uint64_t> const &));

template DLL_PUBLIC void Array<double>::sort_function(
    Array<uint64_t> &, bool (*)(std::pair<double, uint64_t> const &,
                                std::pair<double, uint64_t> const &));

template DLL_PUBLIC void Array<int>::sort_function(
    Array<uint64_t> &, bool (*)(std::pair<int, uint64_t> const &,
                                std::pair<int, uint64_t> const &));

template DLL_PUBLIC void Array<int64_t>::sort_function(
    Array<uint64_t> &, bool (*)(std::pair<int64_t, uint64_t> const &,
                                std::pair<int64_t, uint64_t> const &));

template DLL_PUBLIC void Array<unsigned int>::sort_function(
    Array<uint64_t> &, bool (*)(std::pair<unsigned int, uint64_t> const &,
                                std::pair<unsigned int, uint64_t> const &));

template DLL_PUBLIC void Array<unsigned short>::sort_function(
    Array<uint64_t> &, bool (*)(std::pair<unsigned short, uint64_t> const &,
                                std::pair<unsigned short, uint64_t> const &));

template DLL_PUBLIC void Array<short>::sort_function(
    Array<uint64_t> &, bool (*)(std::pair<short, uint64_t> const &,
                                std::pair<short, uint64_t> const &));

template DLL_PUBLIC void Array<uint64_t>::sort_function(
    Array<uint64_t> &, bool (*)(std::pair<uint64_t, uint64_t> const &,
                                std::pair<uint64_t, uint64_t> const &));

#if defined(_MSC_VER) || defined(__APPLE__)

template DLL_PUBLIC void Array<std::atomic<float>>::sort_function(
    Array<unsigned long long> &,
    bool (*)(std::pair<std::atomic<float>, unsigned long long> const &,
             std::pair<std::atomic<float>, unsigned long long> const &));

template DLL_PUBLIC void Array<std::atomic<double>>::sort_function(
    Array<unsigned long long> &,
    bool (*)(std::pair<std::atomic<double>, unsigned long long> const &,
             std::pair<std::atomic<double>, unsigned long long> const &));

#elif defined(__INTEL_COMPILER)

template DLL_PUBLIC void Array<std::atomic<float> >::sort_function(
    Array<uint64_t> &,
    bool (*)(std::pair<std::atomic<float>, uint64_t> const &,
             std::pair<std::atomic<float>, uint64_t> const &));

template DLL_PUBLIC void Array<std::atomic<double> >::sort_function(
    Array<uint64_t> &,
    bool (*)(std::pair<std::atomic<double>, uint64_t> const &,
             std::pair<std::atomic<double>, uint64_t> const &));

#endif

template void Array<double>::SET_DATA_INDEX<double>(Array<double> &arr,
                                                    size_t index, double value);

template void Array<double>::SET_T_FROM_K<double>(double &t, double k);

template DLL_PUBLIC void Array<int>::set_data_index<int>(size_t index,
                                                         int value);
template DLL_PUBLIC void Array<int64_t>::set_data_index<int64_t>(size_t index,
                                                                 int64_t value);

template DLL_PUBLIC void Array<uint64_t>::set_data_index<uint64_t>(
    size_t index, uint64_t value);

template DLL_PUBLIC void Array<short>::set_data_index<short>(size_t index,
                                                             short value);

template DLL_PUBLIC void Array<unsigned short>::set_data_index<unsigned short>(
    size_t index, unsigned short value);

template DLL_PUBLIC void Array<float>::set_data_index<float>(size_t index,
                                                             float value);
template DLL_PUBLIC void Array<double>::set_data_index<double>(size_t index,
                                                               double value);
template DLL_PUBLIC void Array<unsigned int>::set_data_index<unsigned int>(
    size_t index, unsigned int value);

#if defined(_MSC_VER)

template DLL_PUBLIC void Array<std::atomic<float>>::set_data_index<float>(
    unsigned long long, float);
template DLL_PUBLIC void Array<std::atomic<double>>::set_data_index<double>(
    unsigned long long, double);

#endif
