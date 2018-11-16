//
// Created by pwu on 12/19/17.
//

#ifndef TICK_SIMULATION_SRC_HAWKES_CUSTOM_TYPE2_H_
#define  TICK_SIMULATION_SRC_HAWKES_CUSTOM_TYPE2_H_

#include "tick/hawkes/simulation/simu_hawkes.h"

class Hawkes_customType2 : public Hawkes {
 public:
  ulong last_global_n;
  //! @brief the max value of n kept for all f_i(n)
  ulong MaxN;

  double p_chg_at_0;
  double current_num;
  double avg;  //! the normarlizer

  //! extra information used that will be required during simulation of LOB
  std::string simu_mode = "random";
  //! for scenario 1, simulation of LOB

  //! Peng Wu, An array, indicaitng the global status after the i_th event
  VArrayULong global_n;

  //! Explanation of contribution of intensity
  VArrayDouble Total_intensity;
  VArrayDouble Hawkes_intensity;

  SArrayDoublePtrList1D mu_;
  //! array for accelerating the calculation
  ArrayDouble mu_Max;

  ArrayDouble avg_order_size;
  VArrayDouble Qty;  //! exact number of shares in the line

  //! information for extended scenario
  //! scenario name <generate_var_2>
  ArrayDoubleList1D avg_order_size_by_state;

 public:
  Hawkes_customType2(unsigned int n_nodes, int seed, ulong _MaxN,
                     const SArrayDoublePtrList1D &_mu_);

  //! constructor dedicated for simulation
  Hawkes_customType2(unsigned int n_nodes, int seed, ulong _MaxN, const SArrayDoublePtrList1D &_mu_,
                     const ArrayDouble &extrainfo, const std::string _simu_mode);

  // This forbids the unwanted copy of an Hawkes process
  Hawkes_customType2(Hawkes_customType2 &obj) = delete;

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

  VArrayDoublePtr get_Total_intensity() {
    VArrayDoublePtr shared_process = VArrayDouble::new_ptr(Total_intensity);
    return shared_process;
  }

  VArrayDoublePtr get_Hawkes_intensity() {
    VArrayDoublePtr shared_process = VArrayDouble::new_ptr(Hawkes_intensity);
    return shared_process;
  }

  void init_intensity_(ArrayDouble &intensity, double *total_intensity_bound);
};

#endif  // TICK_SIMULATION_SRC_HAWKES_CUSTOM_TYPE2_H_