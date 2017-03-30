
#ifndef TICK_OPTIM_MODEL_SRC_BASE_HAWKES_SINGLE_H_
#define TICK_OPTIM_MODEL_SRC_BASE_HAWKES_SINGLE_H_

#include "hawkes_model.h"

/** \class ModelHawkesSingle
 * \brief Base class of Hawkes models handling only one realization
 */
class ModelHawkesSingle : public ModelHawkes {
 protected:
  //! @brief The process timestamps (a list of arrays)
  SArrayDoublePtrList1D timestamps;

  //! @brief Ending time of the realization
  double end_time;

  //! @brief Number of jumps of the process
  ulong n_total_jumps;

 public:
  //! @brief Constructor
  //! \param max_n_threads : maximum number of threads to be used for multithreading
  //! \param optimization_level : 0 corresponds to no optimization and 1 to use of faster
  //! (approximated) exponential function
  ModelHawkesSingle(const int max_n_threads = 1,
                    const unsigned int optimization_level = 0);

  void set_data(const SArrayDoublePtrList1D &timestamps, const double end_time);

  unsigned int get_n_threads() const;

  friend class ModelHawkesList;
};

#endif  // TICK_OPTIM_MODEL_SRC_BASE_HAWKES_SINGLE_H_
