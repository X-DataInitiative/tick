
#ifndef LIB_INCLUDE_TICK_HAWKES_MODEL_BASE_MODEL_HAWKES_LIST_H_
#define LIB_INCLUDE_TICK_HAWKES_MODEL_BASE_MODEL_HAWKES_LIST_H_

// License: BSD 3 clause

#include "tick/base/base.h"
#include "model_hawkes.h"
#include "model_hawkes_single.h"

/** \class ModelHawkesList
 * \brief Base class of Hawkes models handling several realizations
 */
class DLL_PUBLIC ModelHawkesList : public ModelHawkes {
 protected:
  //! @brief number of given realization (size of timestamps_list)
  ulong n_realizations;

  //! @brief The process timestamps (a list of list of arrays)
  SArrayDoublePtrList2D timestamps_list;

  //! @brief Ending time of the realization
  VArrayDoublePtr end_times = nullptr;

  //! @brief Number of jumps of the process per realization (size=n_realizations)
  VArrayULongPtr n_jumps_per_realization;

 public:
  //! @brief Constructor
  //! \param max_n_threads : number of cores to be used for multithreading. If negative,
  //! the number of physical cores will be used
  //! \param optimization_level : 0 corresponds to no optimization and 1 to use of faster (approximated) exponential function
  ModelHawkesList(const int max_n_threads = 1,
                  const unsigned int optimization_level = 0);

  virtual void set_data(const SArrayDoublePtrList2D &timestamps_list,
                        const VArrayDoublePtr end_times);

  //! @brief returns the number of jumps per realization
  SArrayULongPtr get_n_jumps_per_realization() const {
    return n_jumps_per_realization;
  }

  VArrayDoublePtr get_end_times() const {
    return end_times;
  }

  virtual unsigned int get_n_threads() const;

  SArrayDoublePtrList2D get_timestamps_list() const {
    return timestamps_list;
  }

 public:
  template<class Archive>
  void serialize(Archive &ar) {
    ar(cereal::make_nvp("ModelHawkes", cereal::base_class<ModelHawkes>(this)));

    ar(CEREAL_NVP(n_realizations));
    ar(CEREAL_NVP(timestamps_list));
    ar(CEREAL_NVP(end_times));
    ar(CEREAL_NVP(n_jumps_per_realization));
  }
};

#endif  // LIB_INCLUDE_TICK_HAWKES_MODEL_BASE_MODEL_HAWKES_LIST_H_
