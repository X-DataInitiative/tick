
#ifndef LIB_INCLUDE_TICK_HAWKES_MODEL_BASE_MODEL_HAWKES_H_
#define LIB_INCLUDE_TICK_HAWKES_MODEL_BASE_MODEL_HAWKES_H_

// License: BSD 3 clause

#include "tick/base/base.h"
#include "tick/base_model/model.h"

/** \class ModelHawkes
 * \brief Base class of Hawkes models
 */
class DLL_PUBLIC ModelHawkes : public Model {
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

 public:
  template<class Archive>
  void serialize(Archive &ar) {
    ar(CEREAL_NVP(max_n_threads));
    ar(CEREAL_NVP(optimization_level));
    ar(CEREAL_NVP(weights_computed));
    ar(CEREAL_NVP(n_nodes));
    ar(CEREAL_NVP(n_jumps_per_node));
  }
};

#endif  // LIB_INCLUDE_TICK_HAWKES_MODEL_BASE_MODEL_HAWKES_H_
