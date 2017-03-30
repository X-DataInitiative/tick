#ifndef TICK_OPTIM_MODEL_SRC_BASE_HAWKES_MODEL_H_
#define TICK_OPTIM_MODEL_SRC_BASE_HAWKES_MODEL_H_

#include "base.h"
#include "model.h"

/** \class ModelHawkes
 * \brief Base class of Hawkes models
 */
class ModelHawkes : public Model {
 protected:
  //! @brief Maximum number of threads that will be used for computation
  //! if < 1 then it is set to the maximum number of threads
  unsigned int max_n_threads;

  //! @brief Optimization level.
  //! 0 corresponds to no optimization
  //! 1 corresponds to using faster (approximate) exponential function
  unsigned int optimization_level;

  //! @brief Weather precomputations are up to date of not.
  bool weights_computed;

  //! @brief n_nodes (number of components in the realization)
  ulong n_nodes;

  //! @brief Number of jumps per dimension
  SArrayULongPtr n_jumps_per_node;

 public:
  //! @brief Constructor
  //! \param max_n_threads : maximum number of threads to be used for multithreading
  //! \param optimization_level : 0 corresponds to no optimization and 1 to use of faster
  //! (approximated) exponential function
  ModelHawkes(const int max_n_threads = 1,
              const unsigned int optimization_level = 0);

  //! @brief Returns the number of components of the process
  ulong get_n_nodes() const { return n_nodes; }

  ulong get_n_total_jumps() const { return n_jumps_per_node->sum(); }

  void set_n_threads(const int max_n_threads);

  SArrayULongPtr get_n_jumps_per_node() const { return n_jumps_per_node; }

 protected:
  //! @brief set n_nodes
  void set_n_nodes(const ulong n_nodes);

  //! @brief Custom exponential function taking into account optimization
  //! level
  //! \param x : The value exponential is computed at
  inline double cexp(double x) {
    return optimized_exp(x, optimization_level);
  }

  friend class ModelHawkesList;
};

#endif  // TICK_OPTIM_MODEL_SRC_BASE_HAWKES_MODEL_H_
