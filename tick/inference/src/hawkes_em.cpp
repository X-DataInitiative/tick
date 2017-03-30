
#include "hawkes_em.h"

HawkesEM::HawkesEM(const double kernel_support, const ulong kernel_size,
                   const int max_n_threads)
    : ModelHawkesList(max_n_threads, 0),
      kernel_discretization(nullptr) {
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

void HawkesEM::solve(ArrayDouble &mu, ArrayDouble2d &kernels) {
  if (!weights_computed) allocate_weights();

  if (mu.size() != n_nodes) {
    TICK_ERROR("baseline / mu argument must be an array of size " << n_nodes);
  }
  if (kernels.n_rows() != n_nodes || kernels.n_cols() != n_nodes * kernel_size) {
    TICK_ERROR("kernels argument must be an array of shape ("
                   << n_nodes << ", " << n_nodes * kernel_size << ")");
  }

  // Map
  // Fill next_mu and next_kernels
  next_mu.init_to_zero();
  next_kernels.init_to_zero();
  parallel_run(get_n_threads(), n_nodes * n_realizations,
               &HawkesEM::solve_u_r, this, mu, kernels);

  // Reduce
  // Fill mu and kernels with next_mu and next_kernels
  mu.init_to_zero();
  kernels.init_to_zero();
  for (ulong r = 0; r < n_realizations; r++) {
    for (ulong node_u = 0; node_u < n_nodes; ++node_u) {
      mu[node_u] += view_row(next_mu, r)[node_u];

      ArrayDouble2d next_kernel_u_r(n_nodes, kernel_size,
                                    view_row(next_kernels, r * n_nodes + node_u).data());
      ArrayDouble2d kernel_u(n_nodes, kernel_size, view_row(kernels, node_u).data());
      kernel_u.mult_incr(next_kernel_u_r, 1.);
    }
  }
}

void HawkesEM::solve_u_r(const ulong r_u, const ArrayDouble &mu,
                         ArrayDouble2d &kernels) {
  // Obtain realization and node index from r_u
  const ulong r = static_cast<const ulong>(r_u / n_nodes);
  const ulong node_u = r_u % n_nodes;

  // Fetch corresponding data
  SArrayDoublePtrList1D &realization = timestamps_list[r];
  ArrayDouble2d kernel_u(n_nodes, kernel_size, view_row(kernels, node_u).data());
  const double mu_u = mu[node_u];

  // initialize next data
  ArrayDouble2d next_kernel_ru(n_nodes, kernel_size,
                               view_row(next_kernels, r * n_nodes + node_u).data());
  ArrayDouble2d unnormalized_kernel_ru(n_nodes, kernel_size,
                                       view_row(unnormalized_kernels,
                                                r * n_nodes + node_u).data());
  double &next_mu_ru = view_row(next_mu, r)[node_u];

  ArrayDouble timestamps_u = view(*realization[node_u]);

  // This array will allow us to find quicker the events in each component that
  // have occurred just before the events we will look at
  ArrayULong last_indices(n_nodes);
  for (ulong v = 0; v < n_nodes; v++) {
    last_indices[v] = realization[v]->size();
  }

  // We loop in reverse order to benefit from last_indices
  for (ulong i = timestamps_u.size() - 1; i != static_cast<ulong>(-1); i--) {
    // this array will store temporary values
    unnormalized_kernel_ru.init_to_zero();

    // norm will be equal to mu_u + \sum_v \sum_(t_j < t_i) g_uv(t_i - t_j)
    double norm_u = 0;

    const double t_i = timestamps_u[i];
    for (ulong node_v = 0; node_v < n_nodes; node_v++) {
      ArrayDouble timestamps_v = view(*realization[node_v]);

      // Update the corresponding index such that it is the largest index which
      // satisfies v[index] <= t_i
      while (true) {
        if (last_indices[node_v] == 0) break;
        if (last_indices[node_v] < timestamps_v.size() &&
            t_i >= timestamps_v[last_indices[node_v]])
          break;
        last_indices[node_v]--;
      }
      if (t_i < timestamps_v[last_indices[node_v]]) continue;

      // Get the corresponding kernels and their size
      ArrayDouble kernel_ruv = view_row(kernel_u, node_v);
      ArrayDouble unnormalized_kernel_ruv = view_row(unnormalized_kernel_ru, node_v);

      ulong j0 = last_indices[node_v];
      ulong last_m = 0;

      // So now we loop on the indices of y backward starting from the index computed above
      for (ulong j = j0; j != static_cast<ulong>(-1); j--) {
        // Case the two events are in fact the same one
        if (node_u == node_v && i == j) {
          norm_u += mu_u;
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

            // Update contribution to kernel of p_ij
            unnormalized_kernel_ruv[m] += unnormalized_p_uv_ij;

            // Update the norm
            norm_u += unnormalized_p_uv_ij;
          } else {
            // If the two events are too far away --> we are done with the second loop
            break;
          }
        }
      }
    }
    // We are now ready to perform normalization !

    // If norm is zero then nothing to do (no contribution)
    if (norm_u == 0) continue;

    // Otherwise, we need to norm the kernel_temp's and the mu_temp
    // and add their contributions to the estimation
    next_mu_ru += mu_u / (norm_u * end_times->sum());
    for (ulong node_v = 0; node_v < n_nodes; node_v++) {
      ArrayDouble unnormalized_kernel_ruv = view_row(unnormalized_kernel_ru, node_v);
      ArrayDouble next_kernel_ruv = view_row(next_kernel_ru, node_v);
      for (ulong m = 0; m < kernel_size; m++) {
        const double normalization_term = norm_u * (*n_jumps_per_node)[node_v] * get_kernel_dt(m);
        next_kernel_ruv[m] += unnormalized_kernel_ruv[m] / normalization_term;
      }
    }
  }
}

double HawkesEM::get_kernel_fixed_dt() const {
  if (kernel_discretization == nullptr) {
    return get_kernel_dt();
  } else {
    TICK_ERROR("Cannot get discretization parameter if kernel discretization "
                   "is explicitly set")
  }
}

void HawkesEM::set_kernel_support(const double kernel_support) {
  if (kernel_discretization != nullptr) {
    TICK_ERROR("kernel support cannot be set if kernel discretization "
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
    TICK_ERROR("kernel size cannot be set if kernel discretization "
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
    TICK_ERROR("kernel discretization parameter cannot be set if kernel discretization "
                   "is explicitly set")
  }

  if (kernel_dt <= 0) {
    TICK_ERROR("Kernel discretization parameter must be positive and you have provided "
                   << kernel_dt)
  }
  if (kernel_dt > kernel_support) {
    TICK_ERROR("Kernel discretization parameter must be smaller than kernel support."
                   << "You have provided " << kernel_dt
                   << " and kernel support is " << kernel_support)
  }
  set_kernel_size(static_cast<ulong>(std::ceil(kernel_support / kernel_dt)));
}

void HawkesEM::set_kernel_discretization(const SArrayDoublePtr kernel_discretization1) {
  set_kernel_support(kernel_discretization1->last());
  set_kernel_size(kernel_discretization1->size() - 1);

  // We make a copy as it is a sensitive value (might lead to segfault if modified)
  kernel_discretization = SArrayDouble::new_ptr(kernel_discretization1->size());
  kernel_discretization->mult_fill(*kernel_discretization1, 1.);

  if (kernel_discretization->size() <= 1) {
    TICK_ERROR("Kernel discretization must contain at least two values")
  }

  kernel_discretization->sort();
  weights_computed = false;
}

