#include "tick/preprocessing/longitudinal_preprocessor.h"

template<typename T>
std::vector<std::vector<T>> LongitudinalPreprocessor::split_vector(std::vector<T> array, size_t chunks) {
    if (chunks == 0)
        TICK_ERROR("Chunks size cannot be zero");

    ulong size = array.size();

    ulong closest_multi = size;
    for (; closest_multi%chunks != 0; closest_multi++) {}

    ulong chunk_size = ceil(closest_multi / chunks);

    std::vector<std::vector<T>> out;
    for (ulong i = 0; i < size; i++) {
        ulong new_size = std::min(size - chunk_size * i, chunk_size);
        if (new_size == 0)
            break;
        std::vector<T> tmp(new_size);
        for (ulong j = 0; j < new_size; j++) {
            tmp[j] = array[i*chunk_size+j];
        }
        out.push_back(tmp);
    }
    return out;
}

template DLL_PUBLIC
  std::vector<std::vector<SSparseArrayDouble2dPtr>>
    LongitudinalPreprocessor::split_vector(std::vector<SSparseArrayDouble2dPtr> array, size_t chunks);

template DLL_PUBLIC
  std::vector<std::vector<ArrayDouble2d>>
    LongitudinalPreprocessor::split_vector(std::vector<ArrayDouble2d> array, size_t chunks);

template DLL_PUBLIC
  std::vector<std::vector<ulong>>
    LongitudinalPreprocessor::split_vector(std::vector<ulong> array, size_t chunks);

