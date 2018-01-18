
#ifndef LIB_INCLUDE_TICK_HAWKES_SIMULATION_SIMU_POINT_PROCESS_H_
#define LIB_INCLUDE_TICK_HAWKES_SIMULATION_SIMU_POINT_PROCESS_H_

// License: BSD 3 clause

#include "tick/random/rand.h"
#include "tick/random/rand.h"
#include "tick/array/varray.h"

#include <cereal/types/vector.hpp>

/*! \class PP
 * \brief (Purely virtual) The name of this class stands for Point Processes.
 *  This is the main class that holds a Point Process realization.
 */

class PP {
////////////////////////////////////////////////////////////////////////////////
//                            Attributes
////////////////////////////////////////////////////////////////////////////////

 public:
  /*! @brief A 1-dimensional array of VArrayDoublePtr holding the different
   *  time arrivals of each component of the process
   */
  VArrayDoublePtrList1D timestamps;

 private :
  // Thread safe random generator
  Rand rand;

  // Current time of simulation
  double time;

  // total number of jumps before thinning
  ulong n_total_jumps;

 protected:
  /// @brief the dimension of the point process
  unsigned int n_nodes;

// Intensity related fields
 protected:
  /// @brief Bound of the future total intensity
  double total_intensity_bound;

 private :
  // Current total intensity
  double total_intensity;

  // Current Intensity of each component
  ArrayDouble intensity;

  // Called to init the intensities at start
  void init_intensity();

  // Set to true if negative intensities is encountered
  bool flag_negative_intensity;

  // Keeps track of maximum total intensity bound
  double max_total_intensity_bound;

 protected:
  /// @brief If set then it thresholds negative intensities
  bool threshold_negative_intensity = false;

// Fields to deal with intensity track recording (itr)
 private :
  // The current time for Track recording intensity
  double itr_time;

  // The time step for the track record of the intensity (if negative then
  // no track records)
  double itr_time_step;

  // The track records of the intensity
  VArrayDoublePtrList1D itr;

  // The time corresponding to the track records of the intensity
  VArrayDoublePtr itr_times;

////////////////////////////////////////////////////////////////////////////////
//                            Constructors and destructors
////////////////////////////////////////////////////////////////////////////////
 protected :
  PP() : n_nodes(0) {}

 public:
  /// @brief Constructor
  /// \param n_nodes : Dimension of the process to be simulated
  explicit PP(unsigned int n_nodes, int seed = -1);

  /// Destructor
  virtual ~PP();

////////////////////////////////////////////////////////////////////////////////
//                            Methods
////////////////////////////////////////////////////////////////////////////////

 public :
  /**
   * @brief Builds the process up to time endTime
   * \param end_time : Time until the realization is performed
   */
  void simulate(double end_time);

  /**
   * @brief Builds the process until the given number of points is reached
   * \param n_points : The number of points until we keep simulating
   * \warning This introduces a small biais especially if the number of points
   * is small
   */
  void simulate(ulong n_points);

  /**
   * @brief Builds the process up to time endTime and stops if the number of
   * points is reached
   * \param end_time : Time until the realization is performed
   * \param n_points : The number of points until we keep simulating
   * \warning This introduces a small biais especially if the number of points
   * is small
   */
  void simulate(double end_time, ulong n_points);

  /**
   * @brief Resets the process as if it was never simulated
   */
  virtual void reset();

  /**
   * @brief Set timestamps of a new point process
   * @param timestamps : timestamps that will be set
   * @param end_time : end_time corresponding to these timestamps
   */
  void set_timestamps(VArrayDoublePtrList1D &timestamps, double end_time);

  /**
   * @brief (Des)Activate track recording of intensity
   * @param dt : The time step used for track recording the intensity (if
   * negative then Desactivate Track Record)
   */
  void activate_itr(double dt);

  /**
   * @brief Reseeds the underlying random generator
   * @param seed New seed for the random generator
   */
  void reseed_random_generator(int seed);

 protected :
  /**
   * @brief Updates the current time so that it goes forward of delay seconds
   * The intensities must be updated and track recorded if needed
   * Returns false if negative intensities were encountered
   * \param delay : Time to update
   * \param intensity : The intensity vector to update
   * \param total_intensity_bound : If not NULL then used to set a bound of
   * total future intensity
   */
  virtual bool update_time_shift_(double delay,
                                  ArrayDouble &intensity,
                                  double *total_intensity_bound) { return false; }

  /**
   * @brief Record a jump in ith component
   */
  void update_jump(int index);

 private :
  /**
   * @brief Update a time shift of delay seconds and eventually recompute the
   * intensity bound if asked and update track record of intensity if asked
   */
  void update_time_shift(double delay,
                         bool flag_compute_intensity_bound,
                         bool flag_itr);

  /**
   * @brief Process track record of intensity at current time
   */
  // TODO: Running with this is slower (30%) than the original library
  void itr_process();

 protected:
  /**
   * @brief Virtual method called once (at startup) to set the initial
   * intensity
   * \param intensity : The intensity vector (of size #dimension) to initialize
   * \param total_intensity_bound : A pointer to the variable that will hold a
   * bound of future total intensity
   */
  virtual void init_intensity_(ArrayDouble &intensity,
                               double *total_intensity_bound);


////////////////////////////////////////////////////////////////////////////////
//                            Getters and setters
////////////////////////////////////////////////////////////////////////////////

 public:
  /// @brief Returns the dimension of the process
  inline unsigned int get_n_nodes() { return n_nodes; }

  /// @brief Returns current time of simulation
  inline double get_time() { return time; }

  /// @brief Returns total number of jumps
  inline ulong get_n_total_jumps() { return n_total_jumps; }

  /// @brief Returns seed of random generator
  int get_seed() const { return rand.get_seed(); }

  /// @brief Returns intensity track record array
  inline VArrayDoublePtrList1D get_itr() {
    if (!itr_on()) TICK_ERROR("``activate_itr()`` must be call before simulation");

    return itr;
  }

  /// @brief Returns times at which intensity has been recorded
  inline VArrayDoublePtr get_itr_times() {
    if (!itr_on()) TICK_ERROR("``activate_itr()`` must be call before simulation");

    return itr_times;
  }

  /// @brief Returns if we are tracking intensity or not
  inline bool itr_on() { return itr_time_step > 0; }

  /// @brief Returns the step with which we record intensity
  inline double get_itr_step() { return itr_time_step; }

  /// @brief Get the process (converted into fixed size array)
  SArrayDoublePtrList1D get_timestamps() {
    SArrayDoublePtrList1D shared_process =
      std::vector<SArrayDoublePtr>(timestamps.begin(), timestamps.end());
    return shared_process;
  }

  /// @brief Gets Maximimum Total intensity bound that wwas encountered during realization
  inline double get_max_total_intensity_bound() { return max_total_intensity_bound; }

  bool get_threshold_negative_intensity() const {
    return threshold_negative_intensity;
  }

  void set_threshold_negative_intensity(const bool threshold_negative_intensity) {
    this->threshold_negative_intensity = threshold_negative_intensity;
  }

////////////////////////////////////////////////////////////////////////////////
//                            Serialization
////////////////////////////////////////////////////////////////////////////////

  template<class Archive>
  void load(Archive &ar) {
    ar(CEREAL_NVP(timestamps));
    ar(CEREAL_NVP(time));
    ar(CEREAL_NVP(n_total_jumps));
    ar(CEREAL_NVP(n_nodes));
    ar(CEREAL_NVP(total_intensity_bound));
    ar(CEREAL_NVP(total_intensity));
    ar(CEREAL_NVP(intensity));
    ar(CEREAL_NVP(flag_negative_intensity));
    ar(CEREAL_NVP(max_total_intensity_bound));
    ar(CEREAL_NVP(threshold_negative_intensity));
    ar(CEREAL_NVP(itr_time));
    ar(CEREAL_NVP(itr_time_step));
    ar(CEREAL_NVP(itr));
    ar(CEREAL_NVP(itr_times));

    int rand_seed;
    ar(CEREAL_NVP(rand_seed));

    rand = Rand(rand_seed);
  }

  template<class Archive>
  void save(Archive &ar) const {
    ar(CEREAL_NVP(timestamps));
    ar(CEREAL_NVP(time));
    ar(CEREAL_NVP(n_total_jumps));
    ar(CEREAL_NVP(n_nodes));
    ar(CEREAL_NVP(total_intensity_bound));
    ar(CEREAL_NVP(total_intensity));
    ar(CEREAL_NVP(intensity));
    ar(CEREAL_NVP(flag_negative_intensity));
    ar(CEREAL_NVP(max_total_intensity_bound));
    ar(CEREAL_NVP(threshold_negative_intensity));
    ar(CEREAL_NVP(itr_time));
    ar(CEREAL_NVP(itr_time_step));
    ar(CEREAL_NVP(itr));
    ar(CEREAL_NVP(itr_times));

    // Note that only the seed is part of the serialization.
    //
    // If the generator has been used (i.e. numbers have been drawn from it) this will not be
    // reflected in the restored (deserialized) object.
    const auto rand_seed = rand.get_seed();
    ar(CEREAL_NVP(rand_seed));
  }
};

#endif  // LIB_INCLUDE_TICK_HAWKES_SIMULATION_SIMU_POINT_PROCESS_H_
