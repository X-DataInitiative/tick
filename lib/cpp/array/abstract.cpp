

#include "tick/array/array.h"

template <class T>
template <class K>
K AbstractArray1d2d<T>::get_data_index(size_t index) const {
  return _data[index];
}

template <>
template <>
float AbstractArray1d2d<std::atomic<float>>::get_data_index(
    size_t index) const {
  return _data[index].load();
}

template <>
template <>
double AbstractArray1d2d<std::atomic<double>>::get_data_index(
    size_t index) const {
  return _data[index].load();
}

template <typename T>
template <typename K>
void AbstractArray1d2d<T>::multiply(const K a) {
  if (_size == 0) TICK_ERROR("Cannot apply *= on an empty array");
  if (size_data() == 0) return;

  tick::vector_operations<T>{}.template scale<K>(size_data(), a, _data);
}
template <>
template <>
void AbstractArray1d2d<std::atomic<float> >::multiply<float>(const float a) {
  if (_size == 0) TICK_ERROR("Cannot apply *= on an empty array");
  if (size_data() == 0) return;

  tick::vector_operations<std::atomic<float>>{}.template scale<float>(
      size_data(), a, _data);
}

template <>
template <>
void AbstractArray1d2d<std::atomic<double>>::multiply<double>(const double a) {
  if (_size == 0) TICK_ERROR("Cannot apply *= on an empty array");
  if (size_data() == 0) return;

  tick::vector_operations<std::atomic<double>>{}.template scale<double>(
      size_data(), a, _data);
}

template class AbstractArray1d2d<unsigned long>;
template class AbstractArray1d2d<float>;
template class AbstractArray1d2d<double>;
template class AbstractArray1d2d<short>;
template class AbstractArray1d2d<unsigned short>;
template class AbstractArray1d2d<unsigned int>;
template class AbstractArray1d2d<long>;

template DLL_PUBLIC unsigned long
    AbstractArray1d2d<unsigned long>::get_data_index(size_t) const;

template DLL_PUBLIC float AbstractArray1d2d<float>::get_data_index<float>(
    size_t) const;

template DLL_PUBLIC double AbstractArray1d2d<double>::get_data_index<double>(
    size_t) const;

template DLL_PUBLIC short AbstractArray1d2d<short>::get_data_index<short>(
    size_t) const;

template DLL_PUBLIC unsigned short AbstractArray1d2d<
    unsigned short>::get_data_index<unsigned short>(size_t) const;

template DLL_PUBLIC int AbstractArray1d2d<int>::get_data_index<int>(
    size_t) const;

template DLL_PUBLIC unsigned int
    AbstractArray1d2d<unsigned int>::get_data_index<unsigned int>(size_t) const;

template DLL_PUBLIC long AbstractArray1d2d<long>::get_data_index<long>(
    size_t) const;

template DLL_PUBLIC long long
    AbstractArray1d2d<long long>::get_data_index<long long>(size_t) const;

template DLL_PUBLIC unsigned long long AbstractArray1d2d<
    unsigned long long>::get_data_index<unsigned long long>(size_t) const;

#if defined(_MSC_VER)
template DLL_PUBLIC double AbstractArray1d2d<
    std::atomic<double>>::get_data_index<double>(size_t) const;

template DLL_PUBLIC float
    AbstractArray1d2d<std::atomic<float>>::get_data_index<float>(size_t) const;
#endif
