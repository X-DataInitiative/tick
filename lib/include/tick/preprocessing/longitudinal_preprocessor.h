
#ifndef LIB_INCLUDE_TICK_PREPROCESSING_LONGITUDINAL_PREPROCESSOR_H_
#define LIB_INCLUDE_TICK_PREPROCESSING_LONGITUDINAL_PREPROCESSOR_H_

#include <cereal/types/base_class.hpp>
#include <cereal/types/polymorphic.hpp>
#include "tick/base/base.h"
#include "tick/base/serialization.h"

class DLL_PUBLIC LongitudinalPreprocessor {
 protected:
  size_t n_jobs;

 public:
  explicit LongitudinalPreprocessor(size_t n_jobs) {
    if (n_jobs == 0)
      this->n_jobs = std::thread::hardware_concurrency();
    else
      this->n_jobs = n_jobs;
  }

  template<typename T>
  std::vector<std::vector<T>> split_vector(std::vector<T> array, size_t chunks);

  template <class Archive>
  void load(Archive &ar) {
    ar(CEREAL_NVP(n_jobs));
  }

  template <class Archive>
  void save(Archive &ar) const {
    ar(CEREAL_NVP(n_jobs));
  }
};

#endif  // LIB_INCLUDE_TICK_PREPROCESSING_LONGITUDINAL_PREPROCESSOR_H_
