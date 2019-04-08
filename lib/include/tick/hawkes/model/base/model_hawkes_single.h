
#ifndef LIB_INCLUDE_TICK_HAWKES_MODEL_BASE_MODEL_HAWKES_SINGLE_H_
#define LIB_INCLUDE_TICK_HAWKES_MODEL_BASE_MODEL_HAWKES_SINGLE_H_

// License: BSD 3 clause

#include <cereal/types/base_class.hpp>
#include <cereal/types/polymorphic.hpp>
#include <cereal/types/vector.hpp>

#include "model_hawkes.h"

/** \class ModelHawkesSingle
 * \brief Base class of Hawkes models handling only one realization
 */
class DLL_PUBLIC ModelHawkesSingle : public ModelHawkes {
 protected:
  //! @brief The process timestamps (a list of arrays)
  SArrayDoublePtrList1D timestamps;

  //! @brief Ending time of the realization
  double end_time;

  //! @brief Number of jumps of the process
  ulong n_total_jumps;

 public:
  //! @brief Constructor
  //! \param max_n_threads : maximum number of threads to be used for
  //! multithreading \param optimization_level : 0 corresponds to no
  //! optimization and 1 to use of faster (approximated) exponential function
  ModelHawkesSingle(const int max_n_threads = 1,
                    const unsigned int optimization_level = 0);

  void set_data(const SArrayDoublePtrList1D &timestamps, const double end_time);

  unsigned int get_n_threads() const;

  double get_end_time() const { return end_time; }

  friend class ModelHawkesList;

 public:
  template <class Archive>
  void serialize(Archive &ar) {
    ar(cereal::make_nvp("ModelHawkes", cereal::base_class<ModelHawkes>(this)));

    ar(CEREAL_NVP(timestamps));
    ar(CEREAL_NVP(end_time));
    ar(CEREAL_NVP(n_total_jumps));
  }

  BoolStrReport compare(const ModelHawkesSingle &that, std::stringstream &ss) {
    ss << get_class_name() << std::endl;
    auto are_equal = ModelHawkes::compare(that, ss) &&
                     TICK_CMP_REPORT_VECTOR_SPTR_1D(ss, timestamps, double) &&
                     TICK_CMP_REPORT(ss, end_time) &&
                     TICK_CMP_REPORT(ss, n_total_jumps);
    return BoolStrReport(are_equal, ss.str());
  }
  BoolStrReport compare(const ModelHawkesSingle &that) {
    std::stringstream ss;
    return compare(that, ss);
  }
  BoolStrReport operator==(const ModelHawkesSingle &that) {
    return ModelHawkesSingle::compare(that);
  }
};

CEREAL_REGISTER_TYPE(ModelHawkesSingle);

#endif  // LIB_INCLUDE_TICK_HAWKES_MODEL_BASE_MODEL_HAWKES_SINGLE_H_
