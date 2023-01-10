// License: BSD 3 clause

#include "tick/hawkes/inference/hawkes_em.h"

HawkesEM::HawkesEM(const double kernel_support, const ulong kernel_size, const int max_n_threads)
    : ModelHawkesList(max_n_threads, 0), kernel_discretization(nullptr) {
  set_kernel_support(kernel_support);
  set_kernel_size(kernel_size);
}

HawkesEM::HawkesEM(const SArrayDoublePtr kernel_discretization, const int max_n_threads)
    : ModelHawkesList(max_n_threads, 0) {
  set_kernel_discretization(kernel_discretization);
}

void HawkesEM::allocate_weights() {
  next_mu = ArrayDouble2d(n_realizations, n_nodes);
  next_kernels = ArrayDouble2d(n_realizations * n_nodes, n_nodes * kernel_size);
  unnormalized_kernels = ArrayDouble2d(n_realizations * n_nodes, n_nodes * kernel_size);
  weights_computed = true;
}

double HawkesEM::loglikelihood(const ArrayDouble &mu, ArrayDouble2d &kernels) {
  check_baseline_and_kernels(mu, kernels);

  double llh = parallel_map_additive_reduce(get_n_threads(), n_nodes * n_realizations,
                                            &HawkesEM::loglikelihood_ur, this, mu, kernels);
  return llh /= get_n_total_jumps();
}

void HawkesEM::solve(ArrayDouble &mu, ArrayDouble2d &kernels) {
  if (!weights_computed) allocate_weights();
  check_baseline_and_kernels(mu, kernels);

  // Map
  // Fill next_mu and next_kernels
  next_mu.init_to_zero();
  next_kernels.init_to_zero();
  parallel_run(get_n_threads(), n_nodes * n_realizations, &HawkesEM::solve_ur, this, mu, kernels);

  // Reduce
  // Fill mu and kernels with next_mu and next_kernels
  mu.init_to_zero();
  kernels.init_to_zero();
  for (ulong r = 0; r < n_realizations; r++) {
    for (ulong node_u = 0; node_u < n_nodes; ++node_u) {
      mu[node_u] += next_mu(r, node_u);

      ArrayDouble2d next_kernel_u_r(n_nodes, kernel_size,
                                    view_row(next_kernels, r * n_nodes + node_u).data());
      ArrayDouble2d kernel_u(n_nodes, kernel_size, view_row(kernels, node_u).data());
      kernel_u.mult_incr(next_kernel_u_r, 1.);
    }
  }
}

double HawkesEM::loglikelihood_ur(const ulong r_u, const ArrayDouble &mu, ArrayDouble2d &kernels) {
  const ulong r = static_cast<const ulong>(r_u / n_nodes);
  double llh = (*end_times)[r];
  std::function<void(double)> add_to_llh = [&llh](double intensity_t_i) {
    if (intensity_t_i <= 0)
      llh = std::numeric_limits<double>::infinity();
    else
      llh += log(intensity_t_i);
  };

  compute_intensities_ur(r_u, mu, kernels, add_to_llh, false);

  llh -= compute_compensator_ur(r_u, mu, kernels);
  return llh;
}

void HawkesEM::solve_ur(const ulong r_u, const ArrayDouble &mu, ArrayDouble2d &kernels) {
  // Obtain realization and node index from r_u
  const ulong r = static_cast<const ulong>(r_u / n_nodes);
  const ulong node_u = r_u % n_nodes;

  // Fetch corresponding data
  const double mu_u = mu[node_u];

  // initialize next data
  ArrayDouble2d next_kernel_ru(n_nodes, kernel_size,
                               view_row(next_kernels, r * n_nodes + node_u).data());
  ArrayDouble2d unnormalized_kernel_ru(n_nodes, kernel_size,
                                       view_row(unnormalized_kernels, r * n_nodes + node_u).data());
  double &next_mu_ru = next_mu(r, node_u);

  std::function<void(double)> add_to_next_kernel = [this, &unnormalized_kernel_ru, &next_kernel_ru,
                                                    &next_mu_ru, &mu_u](double intensity_t_i) {
    // If norm is zero then nothing to do (no contribution)
    if (intensity_t_i == 0) return;

    // Otherwise, we need to norm the kernel_temp's and the mu_temp
    // and add their contributions to the estimation
    next_mu_ru += mu_u / (intensity_t_i * this->end_times->sum());
    for (ulong node_v = 0; node_v < this->n_nodes; node_v++) {
      ArrayDouble unnormalized_kernel_ruv = view_row(unnormalized_kernel_ru, node_v);
      ArrayDouble next_kernel_ruv = view_row(next_kernel_ru, node_v);
      for (ulong m = 0; m < this->kernel_size; m++) {
        const double normalization_term =
            intensity_t_i * (*this->n_jumps_per_node)[node_v] * get_kernel_dt(m);
        next_kernel_ruv[m] += unnormalized_kernel_ruv[m] / normalization_term;
      }
    }
  };
  compute_intensities_ur(r_u, mu, kernels, add_to_next_kernel, true);
}

SArrayDouble2dPtr HawkesEM::get_kernel_norms(ArrayDouble2d &kernels) const {
  check_baseline_and_kernels(ArrayDouble(n_nodes), kernels);

  ArrayDouble discretization_intervals(kernel_size);
  for (ulong m = 0; m < kernel_size; ++m) {
    discretization_intervals[m] = get_kernel_dt(m);
  }

  ArrayDouble2d kernel_norms(n_nodes, n_nodes);
  for (ulong node_u = 0; node_u < n_nodes; ++node_u) {
    ArrayDouble2d kernel_u(n_nodes, kernel_size, view_row(kernels, node_u).data());
    for (ulong node_v = 0; node_v < n_nodes; ++node_v) {
      ArrayDouble kernel_uv = view_row(kernel_u, node_v);
      kernel_norms(node_u, node_v) = kernel_uv.dot(discretization_intervals);
    }
  }

  return kernel_norms.as_sarray2d_ptr();
}

SArrayDouble2dPtr HawkesEM::get_kernel_primitives(ArrayDouble2d &kernels) const {
  check_baseline_and_kernels(ArrayDouble(n_nodes), kernels);

  ArrayDouble discretization_intervals(kernel_size);
  for (ulong m = 0; m < kernel_size; ++m) {
    discretization_intervals[m] = get_kernel_dt(m);
  }

  ArrayDouble2d vals(n_nodes, n_nodes * kernel_size);
  for (ulong u = 0; u < n_nodes; ++u) {
    ArrayDouble2d kernel_u(n_nodes, kernel_size, view_row(kernels, u).data());
    for (ulong v = 0; v < n_nodes; ++v) {
      ArrayDouble kernel_uv = view_row(kernel_u, v);
      double val = 0.;
      for (ulong k = 0; k < kernel_size; ++k) {
        ulong vk = v * kernel_size + k;
        double increment = kernel_uv[k] * discretization_intervals[k];
        if (increment < 0) {
          throw std::runtime_error(
              "Error in computing primitive of the kernel: increment is negative!");
        }
        val += increment;
        vals(u, vk) = val;
      }
    }
  }

  return vals.as_sarray2d_ptr();
}

void HawkesEM::init_kernel_time_func(ArrayDouble2d &kernels) {
  // `kernels` is expected in the shape (n_nodes, n_nodes * kernel_size)
  kernel_time_func.clear();
  kernel_time_func.reserve(n_nodes * n_nodes);
  ArrayDouble abscissa(kernel_size);
  for (ulong t = 0; t < kernel_size; ++t) abscissa[t] = (*kernel_discretization)[t + 1];
  for (ulong u = 0; u < n_nodes; ++u) {
    ArrayDouble2d kernel_u(n_nodes, kernel_size, view_row(kernels, u).data());
    for (ulong v = 0; v < n_nodes; ++v) {
      ArrayDouble kernel_uv = view_row(kernel_u, v);
      kernel_time_func[u * n_nodes + v] =
          TimeFunction(abscissa, kernel_uv, TimeFunction::BorderType::Border0,
                       TimeFunction::InterMode::InterConstLeft);
    }
  }
  is_kernel_time_func = 1;
}

void HawkesEM::compute_intensities_ur(const ulong r_u, const ArrayDouble &mu,
                                      ArrayDouble2d &kernels,
                                      std::function<void(double)> intensity_func,
                                      bool store_unnormalized_kernel) {
  // Obtain realization and node index from r_u
  const ulong r = static_cast<const ulong>(r_u / n_nodes);
  const ulong node_u = r_u % n_nodes;

  ArrayDouble2d kernel_norms = *get_kernel_norms(kernels);
  // ?
  // double marginal_int_intensity = 0;
  // for (ulong node_v = 0; node_v < n_nodes; ++node_v) {
  //   marginal_int_intensity += kernel_norms(node_v, node_u);
  // }

  // Fetch corresponding data
  SArrayDoublePtrList1D &realization = timestamps_list[r];
  ArrayDouble2d kernel_u(n_nodes, kernel_size, view_row(kernels, node_u).data());
  const double mu_u = mu[node_u];

  ArrayDouble2d unnormalized_kernel_ru;
  if (store_unnormalized_kernel) {
    unnormalized_kernel_ru = ArrayDouble2d(
        n_nodes, kernel_size, view_row(unnormalized_kernels, r * n_nodes + node_u).data());
  }

  ArrayDouble timestamps_u = view(*realization[node_u]);

  // This array will allow us to find quicker the events in each component that
  // have occurred just before the events we will look at
  ArrayULong last_indices(n_nodes);
  for (ulong v = 0; v < n_nodes; v++) {
    last_indices[v] = realization[v]->size();
  }

  for (ulong i = timestamps_u.size() - 1; i != static_cast<ulong>(-1); i--) {
    const double t_i = timestamps_u[i];
    unnormalized_kernel_ru.init_to_zero();

    // intensity_t_i will be equal to the intensity value of node i at time t_i
    // ie. mu_u + \sum_v \sum_(t_j < t_i) g_uv(t_i - t_j)
    double intensity_t_i = 0;
    for (ulong node_v = 0; node_v < n_nodes; node_v++) {
      ArrayDouble timestamps_v = view(*realization[node_v]);

      ArrayDouble unnormalized_kernel_ruv;
      if (store_unnormalized_kernel)
        unnormalized_kernel_ruv = view_row(unnormalized_kernel_ru, node_v);

      // Update the corresponding index such that it is the largest index which
      // satisfies v[index] <= t_i
      while (true) {
        if (last_indices[node_v] == 0) break;
        if (last_indices[node_v] < timestamps_v.size() && t_i >= timestamps_v[last_indices[node_v]])
          break;
        last_indices[node_v]--;
      }
      // if (t_i < timestamps_v[last_indices[node_v]]) continue;
      if ((timestamps_v.size() == 0) || (t_i < timestamps_v[last_indices[node_v]])) continue;

      // Get the corresponding kernels and their size
      ArrayDouble kernel_ruv = view_row(kernel_u, node_v);

      ulong j0 = last_indices[node_v];
      ulong last_m = 0;

      // So now we loop on the indices of y backward starting from the index
      // computed above
      for (ulong j = j0; j != static_cast<ulong>(-1); j--) {
        // Case the two events are in fact the same one
        if (node_u == node_v && i == j) {
          intensity_t_i += mu_u;
        } else {
          // Case they are different
          const double t_j = timestamps_v[j];

          const double t_diff = t_i - t_j;
          if (t_diff < kernel_support) {
            // We get the index in the kernel array
            ulong m;
            if (kernel_discretization == nullptr) {
              m = static_cast<ulong>(floor(t_diff / get_kernel_dt()));
            } else {
              // last_m allows us to find m value quicker as m >= last_m
              m = last_m;
              while ((*kernel_discretization)[m + 1] < t_diff) m++;
            }
            last_m = m;

            // Then we get the corresponding kernel value
            double unnormalized_p_uv_ij = kernel_ruv[m];

            if (store_unnormalized_kernel) unnormalized_kernel_ruv[m] += unnormalized_p_uv_ij;

            // Update the norm
            intensity_t_i += unnormalized_p_uv_ij;
          } else {
            // If the two events are too far away --> we are done with the
            // second loop
            break;
          }
        }
      }
    }
    intensity_func(intensity_t_i);
  }
}

double HawkesEM::compute_compensator_ur(const ulong r_u, const ArrayDouble &mu,
                                        ArrayDouble2d &kernels) {
  // Obtain realization and node index from r_u
  const ulong r = static_cast<const ulong>(r_u / n_nodes);
  const ulong node_u = r_u % n_nodes;

  double compensator = 0;
  ArrayDouble2d kernel_norms = *get_kernel_norms(kernels);

  // Marginal value added to compensator by an event of node u
  double marginal_compensator = 0;
  for (ulong node_v = 0; node_v < n_nodes; ++node_v) {
    marginal_compensator += kernel_norms(node_v, node_u);
  }

  // Fetch corresponding data
  SArrayDoublePtrList1D &realization = timestamps_list[r];
  const double mu_u = mu[node_u];

  ArrayDouble timestamps_u = view(*realization[node_u]);

  // Marginal compensator of the kernel u for timestamps are close to end_time
  double current_marginal_compensator = 0;
  ulong last_m = 0;
  ArrayDouble kernel_discretization = *get_kernel_discretization();
  for (ulong i = timestamps_u.size() - 1; i != static_cast<ulong>(-1); i--) {
    const double t_i = timestamps_u[i];
    double t_diff = (*end_times)[r] - t_i;

    // if t_i is too close from end_time its marginal value is less than
    // marginal_compensator
    if (t_diff < kernel_support) {
      // We get the index in the kernel array
      // last_m allows us to find m value quicker as m >= last_m
      ulong m = last_m;
      while (kernel_discretization[m + 1] < t_diff) {
        double interval = kernel_discretization[m + 1] - kernel_discretization[m];
        // We add the compensator value added by bucket m
        for (ulong node_v = 0; node_v < n_nodes; ++node_v) {
          current_marginal_compensator += interval * kernels(node_v, node_u * kernel_size + m);
        }
        m++;
      }
      compensator += current_marginal_compensator;
      double interval_part = t_diff - kernel_discretization[m];
      // We add compensator value of part of the bucket
      for (ulong node_v = 0; node_v < n_nodes; ++node_v) {
        compensator += interval_part * kernels(node_v, node_u * kernel_size + m);
      }
      last_m = m;
    } else {
      // If t is far enough from end_time, its contribution to compensator is
      // marginal_compensator just like all the (i + 1) events that will happen
      // after
      compensator += (i + 1) * marginal_compensator;
      break;
    }
  }

  compensator += mu_u * (*end_times)[r];
  return compensator;
}

SArrayDoublePtr HawkesEM::primitive_of_intensity_at_jump_times(const ulong r_u, ArrayDouble &mu,
                                                               ArrayDouble2d &kernels) {
  _baseline = mu;
  _adjacency = kernels;
  init_kernel_time_func(_adjacency);
  // Obtain realization and node index from r_u
  const ulong r = static_cast<const ulong>(r_u / n_nodes);
  const ulong u = r_u % n_nodes;

  // Fetch corresponding data
  SArrayDoublePtrList1D &realization = timestamps_list[r];
  ArrayDouble timestamps_u = view(*realization[u]);
  ulong n = timestamps_u.size();
  ArrayDouble res = ArrayDouble(n);
  for (ulong k = 0; k < n; ++k) {
    double t_uk = timestamps_u[k];
    res[k] = _evaluate_primitive_of_intensity(t_uk, r, u);
  }
  return res.as_sarray_ptr();
}

double HawkesEM::_evaluate_primitive_of_intensity(const double t, const ulong r, const ulong u) {
  if (is_kernel_time_func == 0) {
    throw std::runtime_error("kernel_time_func has not been initialised yet.");
  }

  // Fetch corresponding data
  SArrayDoublePtrList1D &realization = timestamps_list[r];
  const double mu_u = _baseline[u];

  double res = 0.;
  // Iterate over components and timestamps
  for (ulong v = 0; v < n_nodes; ++v) {
    ArrayDouble timestamps_v = view(*realization[v]);
    ulong n = timestamps_v.size();
    for (ulong k = 0; k < n; k++) {
      double t_vk = timestamps_v[k];
      if (t_vk >= t) break;
      res += kernel_time_func[u * n_nodes + v].primitive(t - t_vk);
    }
  }
  res += mu_u * t;
  return res;
}

double HawkesEM::get_kernel_dt(const ulong m) const {
  if (kernel_discretization == nullptr) {
    return kernel_support / kernel_size;
  } else {
    return (*kernel_discretization)[m + 1] - (*kernel_discretization)[m];
  }
}

void HawkesEM::check_baseline_and_kernels(const ArrayDouble &mu, ArrayDouble2d &kernels) const {
  if (mu.size() != n_nodes) {
    TICK_ERROR("baseline / mu argument must be an array of size " << n_nodes);
  }
  if (kernels.n_rows() != n_nodes || kernels.n_cols() != n_nodes * kernel_size) {
    TICK_ERROR("kernels argument must be an array of shape (" << n_nodes << ", "
                                                              << n_nodes * kernel_size << ")");
  }
}

double HawkesEM::get_kernel_fixed_dt() const {
  if (kernel_discretization == nullptr) {
    return get_kernel_dt();
  } else {
    TICK_ERROR(
        "Cannot get discretization parameter if kernel discretization "
        "is explicitly set")
  }
}

void HawkesEM::set_kernel_support(const double kernel_support) {
  if (kernel_discretization != nullptr) {
    TICK_ERROR(
        "kernel support cannot be set if kernel discretization "
        "is explicitly set")
  }
  if (kernel_support <= 0) {
    TICK_ERROR("Kernel support must be positive and you have provided " << kernel_support)
  }
  this->kernel_support = kernel_support;
  weights_computed = false;
}

void HawkesEM::set_kernel_size(const ulong kernel_size) {
  if (kernel_discretization != nullptr) {
    TICK_ERROR(
        "kernel size cannot be set if kernel discretization "
        "is explicitly set")
  }
  if (kernel_size <= 0) {
    TICK_ERROR("Kernel size must be positive and you have provided " << kernel_size)
  }
  this->kernel_size = kernel_size;
  weights_computed = false;
}

void HawkesEM::set_kernel_dt(const double kernel_dt) {
  if (kernel_discretization != nullptr) {
    TICK_ERROR(
        "kernel discretization parameter cannot be set if kernel "
        "discretization "
        "is explicitly set")
  }

  if (kernel_dt <= 0) {
    TICK_ERROR(
        "Kernel discretization parameter must be positive and you have "
        "provided "
        << kernel_dt)
  }
  if (kernel_dt > kernel_support) {
    TICK_ERROR("Kernel discretization parameter must be smaller than kernel support."
               << "You have provided " << kernel_dt << " and kernel support is " << kernel_support)
  }
  set_kernel_size(static_cast<ulong>(std::ceil(kernel_support / kernel_dt)));
}

void HawkesEM::set_kernel_discretization(const SArrayDoublePtr kernel_discretization1) {
  set_kernel_support(kernel_discretization1->last());
  set_kernel_size(kernel_discretization1->size() - 1);

  // We make a copy as it is a sensitive value (might lead to segfault if
  // modified)
  kernel_discretization = SArrayDouble::new_ptr(kernel_discretization1->size());
  kernel_discretization->mult_fill(*kernel_discretization1, 1.);

  if (kernel_discretization->size() <= 1) {
    TICK_ERROR("Kernel discretization must contain at least two values")
  }

  kernel_discretization->sort();
  weights_computed = false;
}

SArrayDoublePtr HawkesEM::get_kernel_discretization() const {
  if (kernel_discretization == nullptr) {
    ArrayDouble kernel_discretization_tmp = arange<double>(0, kernel_size + 1);
    kernel_discretization_tmp.mult_fill(kernel_discretization_tmp, get_kernel_fixed_dt());
    return kernel_discretization_tmp.as_sarray_ptr();
  } else {
    return kernel_discretization;
  }
}
