
#ifndef LIB_INCLUDE_TICK_HAWKES_MODEL_BASE_MODEL_HAWKES_LIST_H_
#define LIB_INCLUDE_TICK_HAWKES_MODEL_BASE_MODEL_HAWKES_LIST_H_

// License: BSD 3 clause

#include "model_hawkes.h"
#include "model_hawkes_single.h"
#include "tick/base/base.h"

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

  //! @brief Number of jumps of the process per realization
  //! (size=n_realizations)
  VArrayULongPtr n_jumps_per_realization;

 public:
  //! @brief Constructor
  //! \param max_n_threads : number of cores to be used for multithreading. If
  //! negative, the number of physical cores will be used \param
  //! optimization_level : 0 corresponds to no optimization and 1 to use of
  //! faster (approximated) exponential function
  ModelHawkesList(const int max_n_threads = 1, const unsigned int optimization_level = 0);

  virtual void set_data(const SArrayDoublePtrList2D &timestamps_list,
                        const VArrayDoublePtr end_times);

  //! @brief returns the number of jumps per realization
  SArrayULongPtr get_n_jumps_per_realization() const { return n_jumps_per_realization; }

  VArrayDoublePtr get_end_times() const { return end_times; }

  virtual unsigned int get_n_threads() const;

  SArrayDoublePtrList2D get_timestamps_list() const { return timestamps_list; }

 public:
  template <class Archive>
  void load(Archive &ar) {
    ar(cereal::make_nvp("ModelHawkes", cereal::base_class<ModelHawkes>(this)));

    ar(CEREAL_NVP(n_realizations));

    std::vector<std::vector<ArrayDouble> > serialized_timestamps_list;
    ar(CEREAL_NVP(serialized_timestamps_list));

    bool has_end_times = false;
    ar(CEREAL_NVP(has_end_times));
    if (has_end_times) {
      ArrayDouble serialized_end_times;
      ar(cereal::make_nvp("end_times", serialized_end_times));
      end_times = VArrayDouble::new_ptr(serialized_end_times);
    } else {
      end_times = nullptr;
    }

    bool has_n_jumps_per_realization = false;
    ar(CEREAL_NVP(has_n_jumps_per_realization));
    if (has_n_jumps_per_realization) {
      ArrayULong serialized_n_jumps_per_realization;
      ar(cereal::make_nvp("n_jumps_per_realization",
                          serialized_n_jumps_per_realization));
      n_jumps_per_realization =
          VArrayULong::new_ptr(serialized_n_jumps_per_realization);
    } else {
      n_jumps_per_realization = nullptr;
    }

    timestamps_list.clear();
    timestamps_list.reserve(serialized_timestamps_list.size());
    for (auto &serialized_realization : serialized_timestamps_list) {
      SArrayDoublePtrList1D realization;
      realization.reserve(serialized_realization.size());
      for (auto &serialized_timestamps : serialized_realization) {
        realization.push_back(SArrayDouble::new_ptr(serialized_timestamps));
      }
      timestamps_list.push_back(std::move(realization));
    }
  }

  template <class Archive>
  void save(Archive &ar) const {
    ar(cereal::make_nvp("ModelHawkes", cereal::base_class<ModelHawkes>(this)));

    ar(CEREAL_NVP(n_realizations));

    std::vector<std::vector<ArrayDouble> > serialized_timestamps_list;
    serialized_timestamps_list.reserve(timestamps_list.size());
    for (const auto &realization : timestamps_list) {
      std::vector<ArrayDouble> serialized_realization;
      serialized_realization.reserve(realization.size());
      for (const auto &timestamps : realization) {
        serialized_realization.emplace_back(*timestamps);
      }
      serialized_timestamps_list.push_back(std::move(serialized_realization));
    }
    ar(CEREAL_NVP(serialized_timestamps_list));

    const bool has_end_times = end_times != nullptr;
    ar(CEREAL_NVP(has_end_times));
    if (has_end_times) {
      const ArrayDouble serialized_end_times(*end_times);
      ar(cereal::make_nvp("end_times", serialized_end_times));
    }

    const bool has_n_jumps_per_realization = n_jumps_per_realization != nullptr;
    ar(CEREAL_NVP(has_n_jumps_per_realization));
    if (has_n_jumps_per_realization) {
      const ArrayULong serialized_n_jumps_per_realization(
          *n_jumps_per_realization);
      ar(cereal::make_nvp("n_jumps_per_realization",
                          serialized_n_jumps_per_realization));
    }
  }

  BoolStrReport compare(const ModelHawkesList &that, std::stringstream &ss) {
    ss << get_class_name() << std::endl;
    auto are_equal = ModelHawkes::compare(that, ss) && TICK_CMP_REPORT(ss, n_realizations) &&
                     TICK_CMP_REPORT_VECTOR_SPTR_2D(ss, timestamps_list, double) &&
                     TICK_CMP_REPORT_PTR(ss, end_times) &&
                     TICK_CMP_REPORT_PTR(ss, n_jumps_per_realization);
    return BoolStrReport(are_equal, ss.str());
  }
  BoolStrReport compare(const ModelHawkesList &that) {
    std::stringstream ss;
    return compare(that, ss);
  }
  BoolStrReport operator==(const ModelHawkesList &that) { return ModelHawkesList::compare(that); }
};

CEREAL_SPECIALIZE_FOR_ALL_ARCHIVES(ModelHawkesList,
                                   cereal::specialization::member_load_save)

#endif  // LIB_INCLUDE_TICK_HAWKES_MODEL_BASE_MODEL_HAWKES_LIST_H_
