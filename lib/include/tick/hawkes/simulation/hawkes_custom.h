//
// Created by pwu on 12/19/17.
//

#ifndef TICK_SIMULATION_SRC_HAWKES_CUSTOM_H_
#define  TICK_SIMULATION_SRC_HAWKES_CUSTOM_H_

#include "tick/hawkes/simulation/simu_hawkes.h"

class Hawkes_custom : public Hawkes {
 public:
  ulong last_global_n;
  //! @brief the max value of n kept for all f_i(n)
  ulong MaxN_of_f;

  double dim;
  double current_num;
  // the normarlizer
  double avg;

  //! add the additonal information to allow the simulation
  std::string simu_mode = "random";
  //! for scenario 1, simulation of LOB

  //! Peng Wu, An array, indicaitng the global status after the i_th event
  VArrayULong global_n;

  SArrayDoublePtrList1D f_i;
  //! array for accelerating the calculation
  ArrayDouble f_i_Max;
  VArrayDouble Qty;  //! exact number of shares in the line

  ArrayDouble avg_order_size;

 public:
  Hawkes_custom(unsigned int n_nodes, int seed, ulong _MaxN_of_f,
                const SArrayDoublePtrList1D &_f_i);

  Hawkes_custom(unsigned int n_nodes, int seed, ulong _MaxN_of_f, const SArrayDoublePtrList1D &_f_i,
                const ArrayDouble &extrainfo, const std::string _simu_mode = "generate");

  // This forbids the unwanted copy of an Hawkes process
  Hawkes_custom(Hawkes_custom &hawkes_custom) = delete;

  /**
   * @brief Updates the current time so that it goes forward of delay seconds
   * The intensities must be updated and track recorded if needed
   * Returns false if negative intensities were encountered
   * \param delay : Time to update
   * \param intensity : The intensity vector to update
   * \param total_intensity_bound : If not NULL then used to set a bound of
   * total future intensity
   */
  bool update_time_shift_(double delay, ArrayDouble &intensity, double *total_intensity_bound);

  void update_jump(int index);

  VArrayULongPtr get_global_n() {
    VArrayULongPtr shared_process = VArrayULong::new_ptr(global_n);
    return shared_process;
  }

  VArrayDoublePtr get_Qty() {
    VArrayDoublePtr shared_process = VArrayDouble::new_ptr(Qty);
    return shared_process;
  }

  void init_intensity_(ArrayDouble &intensity, double *total_intensity_bound);
};

#endif  // TICK_SIMULATION_SRC_HAWKES_CUSTOM_H_