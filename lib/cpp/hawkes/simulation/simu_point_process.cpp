// License: BSD 3 clause

#include <float.h>
#include "tick/hawkes/simulation/simu_point_process.h"

// Constructor
PP::PP(unsigned int n_nodes, int seed)
  : rand(seed), n_nodes(n_nodes) {
  // Setting the process
  timestamps.resize(n_nodes);
  for (unsigned int i = 0; i < n_nodes; i++) timestamps[i] = VArrayDouble::new_ptr();

  // Init current time
  time = 0;
  itr_time = 0;
  flag_negative_intensity = false;

  max_total_intensity_bound = 0;

  // Init total number of jumps
  n_total_jumps = 0;

//    thresholdNegativeIntensity = false;

  // Initializes intensity vector that will keep current intensity of each component
  intensity = ArrayDouble(n_nodes);
  intensity.init_to_zero();

  // By default : no track record of intensity
  itr_time_step = -1;
}

// Destructor
PP::~PP() {
}

void PP::init_intensity() {
  init_intensity_(intensity, &total_intensity_bound);
  max_total_intensity_bound = total_intensity_bound;
}

void PP::init_intensity_(ArrayDouble &intensity, double *totalIntensityBound) {
  intensity.init_to_zero();

  *totalIntensityBound = 0;
  total_intensity = 0;
}

void PP::reset() {
  time = 0;
  itr_time = 0;
  max_total_intensity_bound = 0;
  n_total_jumps = 0;

  intensity.init_to_zero();

  for (unsigned int i = 0; i < n_nodes; ++i) timestamps[i] = VArrayDouble::new_ptr();
  activate_itr(itr_time_step);
}

void PP::activate_itr(double dt) {
  if (dt <= 0) {
    itr_time_step = -1;
    return;
  }
  if (itr.size() != 0) itr.resize(0);

  itr_time_step = dt;
  itr.resize(n_nodes);
  for (unsigned int i = 0; i < n_nodes; i++) itr[i] = VArrayDouble::new_ptr();
  itr_times = VArrayDouble::new_ptr();
}

void PP::reseed_random_generator(int seed) {
  rand.reseed(seed);
}

void PP::itr_process() {
  if (!itr_on()) return;

  for (unsigned int i = 0; i < n_nodes; i++) itr[i]->append1(intensity[i]);
  itr_times->append1(time);
}

void PP::update_time_shift(double delay, bool flag_compute_intensity_bound, bool flag_itr) {
  flag_negative_intensity = update_time_shift_(delay, intensity,
                                               (flag_compute_intensity_bound
                                                ? &total_intensity_bound : nullptr));

  time += delay;

  if (flag_compute_intensity_bound && max_total_intensity_bound < total_intensity_bound)
    max_total_intensity_bound = total_intensity_bound;

  if (flag_itr) itr_process();
}

void PP::simulate(ulong n_points) {
  simulate(std::numeric_limits<double>::max(), n_points);
}

void PP::simulate(double run_time) {
  simulate(run_time, std::numeric_limits<ulong>::max());
}

void PP::simulate(double end_time, ulong n_points) {
  // This causes deadlock, see MLPP-334 - Investigate deadlock in PP
  // #ifdef PYTHON_LINK
  //    Py_BEGIN_ALLOW_THREADS;
  // #endif

  // At start we need to init the intensity and eventually track record it
  if (get_time() == 0) {
    init_intensity();
    itr_process();
  }

  // We loop till we reach the endTime
  while (time < end_time && n_total_jumps < n_points &&
    (!flag_negative_intensity || threshold_negative_intensity)) {
    // We compute the time of the potential next random jump
    const double time_of_next_jump = time + rand.exponential(total_intensity_bound);

    // If we must track record the intensities we perform a loop
    if (itr_on()) {
      while (itr_time + itr_time_step < std::min(time_of_next_jump, end_time)) {
        update_time_shift(itr_time_step + itr_time - time, false, true);
        if (flag_negative_intensity && !threshold_negative_intensity) break;
        itr_time = itr_time + itr_time_step;
      }
      if (flag_negative_intensity && !threshold_negative_intensity) break;
    }

    // Are we done ?
    if (time_of_next_jump >= end_time) {
      time = end_time;
      break;
    }

    // We go to timeOfNextJump
    // Do not compute intensity bound here as we want new intensities but old bound that we
    // have used for the exponential law
    update_time_shift(time_of_next_jump - time, false, false);
    if (flag_negative_intensity && !threshold_negative_intensity) break;

    // We need to understand which component jumps
    double temp = rand.uniform() * total_intensity_bound;

    unsigned int i;
    for (i = 0; i < n_nodes; i++) {
      temp -= intensity[i];
      if (temp <= 0) break;
    }

    // Case we discard the jump, we should recompute max intensity ?????????  !
    if (i == n_nodes) {
      update_time_shift(0, true, false);
      if (flag_negative_intensity && !threshold_negative_intensity) break;
      continue;
    }

    // Now we are ready to jump if needed
    update_jump(i);

    // We compute the new intensities (taking into account eventually the new jump)
    update_time_shift(0, true, true);

    if (flag_negative_intensity && !threshold_negative_intensity) break;
  }

  // This causes deadlock, see MLPP-334 - Investigate deadlock in PP
  // #ifdef PYTHON_LINK
  //    Py_END_ALLOW_THREADS;
  // #endif

  if (flag_negative_intensity && !threshold_negative_intensity) TICK_ERROR(
    "Simulation stopped because intensity went negative (you could call "
      "``threshold_negative_intensity`` to allow it)");
}

// Update the process component 'index' with current time
void PP::update_jump(int index) {
  // We make the jump on the corresponding signal
  timestamps[index]->append1(time);
  n_total_jumps += 1;
}

void PP::set_timestamps(VArrayDoublePtrList1D &timestamps, double end_time) {
  if (n_nodes != timestamps.size()) {
    TICK_ERROR("Should provide n_nodes (" << n_nodes << ") arrays for timestamps but"
      " was " << timestamps.size());
  }

  reset();

  ArrayULong current_index(n_nodes);
  current_index.init_to_zero();

  const double inf = std::numeric_limits<double>::max();
  while (true) {
    ulong next_jump_node = 0;
    double next_jump_time = inf;

    // Find where is next jump
    for (ulong i = 0; i < n_nodes; ++i) {
      const double next_jump_node_i = current_index[i] < timestamps[i]->size() ?
                                      (*timestamps[i])[current_index[i]] : inf;
      if (next_jump_node_i < next_jump_time) {
        next_jump_node = i;
        next_jump_time = next_jump_node_i;
      }
    }

    // All jumps have been seen
    // We still want to continue to record intensity
    if (next_jump_time == inf) {
      next_jump_time = end_time;
    }

    current_index[next_jump_node]++;

    if (itr_on()) {
      while (itr_time + itr_time_step < next_jump_time) {
        update_time_shift(itr_time_step + itr_time - time, false, true);
        if (flag_negative_intensity && !threshold_negative_intensity) break;
        itr_time = itr_time + itr_time_step;
      }
    }

    // Exit before recording end_time as a jump
    if (next_jump_time == end_time) break;

    update_time_shift(next_jump_time - time, true, true);
    update_jump(next_jump_node);
  }
}
