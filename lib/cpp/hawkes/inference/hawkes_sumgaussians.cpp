// License: BSD 3 clause

#include "tick/base/base.h"
#include "tick/hawkes/inference/hawkes_sumgaussians.h"

// soft-thresholding operator
double soft_thres(double z, double alpha) {
  return (z > 0 ? std::max(std::abs(z) - alpha, 0.) : -std::max(std::abs(z) - alpha, 0.));
}

HawkesSumGaussians::HawkesSumGaussians(const ulong n_gaussians, const double max_mean_gaussian,
                                       const double step_size, const double strength_lasso,
                                       const double strength_grouplasso,
                                       const ulong em_max_iter, const int max_n_threads,
                                       const unsigned int optimization_level)
  : ModelHawkesList(max_n_threads, optimization_level) {
  set_n_gaussians(n_gaussians);
  set_em_max_iter(em_max_iter);
  set_max_mean_gaussian(max_mean_gaussian);
  set_step_size(step_size);
  set_strength_lasso(strength_lasso);
  set_strength_grouplasso(strength_grouplasso);
}

void HawkesSumGaussians::compute_weights() {
  // Allocate weights
  next_mu = ArrayDouble2d(n_realizations, n_nodes);
  next_C = ArrayDouble2d(n_realizations * n_nodes, n_nodes * n_gaussians);
  unnormalized_next_C = ArrayDouble2d(n_realizations * n_nodes, n_nodes * n_gaussians);

  kernel_integral = ArrayDouble(n_nodes * n_gaussians);
  g = ArrayDouble2dList2D(n_realizations);
  for (ulong r = 0; r < n_realizations; r++) {
    g[r] = ArrayDouble2dList1D(n_nodes);
    for (ulong u = 0; u < n_nodes; u++) {
      g[r][u] = ArrayDouble2d(timestamps_list[r][u]->size(), n_nodes * n_gaussians);
    }
  }

  // Compute means_gaussians, std_gaussian and useful constants
  means_gaussians = ArrayDouble(n_gaussians);
  for (ulong m = 0; m < n_gaussians; m++) {
    means_gaussians[m] = static_cast<double>(m) *
      max_mean_gaussian / static_cast<double>(n_gaussians);
  }
  std_gaussian = max_mean_gaussian / (n_gaussians * M_PI);
  std_gaussian_sq = std_gaussian * std_gaussian;
  norm_constant_gauss = std_gaussian * std::sqrt(2. * M_PI);
  norm_constant_erf = std_gaussian * std::sqrt(2);

  // Compute weights
  // variable to compute kernel integral in parallel that will be reduced afterwards
  ArrayDouble2d map_kernel_integral(n_realizations, n_nodes * n_gaussians);
  map_kernel_integral.init_to_zero();
  parallel_run(get_n_threads(), n_nodes * n_realizations, &HawkesSumGaussians::compute_weights_ru,
               this, map_kernel_integral);

  kernel_integral.init_to_zero();
  for (ulong r = 0; r < n_realizations; r++) {
    ArrayDouble map_kernel_integral_r = view_row(map_kernel_integral, r);
    for (ulong u = 0; u < n_nodes; u++) {
      for (ulong m = 0; m < n_gaussians; m++) {
        kernel_integral[u * n_gaussians + m] += map_kernel_integral_r[u * n_gaussians + m];
      }
    }
  }

  weights_computed = true;
}

void HawkesSumGaussians::compute_weights_ru(const ulong r_u, ArrayDouble2d &map_kernel_integral) {
  // Obtain realization and node index from r_u
  const ulong r = static_cast<const ulong>(r_u / n_nodes);
  const ulong u = r_u % n_nodes;
  ArrayDouble2d g_ru = view(g[r][u]);
  g_ru.init_to_zero();
  const ArrayDouble timestamps_ru = view(*timestamps_list[r][u]);
  const double end_time_r = (*end_times)[r];
  ArrayDouble map_kernel_integral_r = view_row(map_kernel_integral, r);

  for (ulong v = 0; v < n_nodes; v++) {
    const ArrayDouble timestamps_rv = view(*timestamps_list[r][v]);
    for (ulong k = 0; k < timestamps_ru.size(); k++) {
      const double t_ru_k = timestamps_ru[k];
      ArrayDouble g_ru_k = view_row(g_ru, k);
      for (ulong m = 0; m < n_gaussians; m++) {
        ulong ij = 0;
        while ((ij < timestamps_rv.size()) && (timestamps_rv[ij] < t_ru_k)) {
          g_ru_k[v * n_gaussians + m] +=
            cexp(-(t_ru_k - timestamps_rv[ij] - means_gaussians[m])
                   * (t_ru_k - timestamps_rv[ij] - means_gaussians[m])
                   / (2. * std_gaussian_sq))
              / norm_constant_gauss;
          ij++;
        }
      }
      if (u == v) {
        // We use this pass over the data to fill kernel_integral
        for (ulong m = 0; m < n_gaussians; m++) {
          map_kernel_integral_r[u * n_gaussians + m] +=
            0.5 * std::erf((end_time_r - t_ru_k - means_gaussians[m]) / norm_constant_erf)
              + 0.5 * std::erf(means_gaussians[m] / norm_constant_erf);
        }
      }
    }
  }
}

// The main method for performing one iteration
void HawkesSumGaussians::solve(ArrayDouble &mu, ArrayDouble2d &amplitudes) {
  if (!weights_computed) compute_weights();

  if (mu.size() != n_nodes) {
    TICK_ERROR("mu argument must be an array of shape (" << n_nodes << ",)");
  }
  if (amplitudes.n_rows() != n_nodes || amplitudes.n_cols() != n_nodes * n_gaussians) {
    TICK_ERROR("amplitudes matrix must be an array of shape (" << n_nodes << ", "
                                                               << n_nodes * n_gaussians << ")");
  }

  ArrayDouble2d amplitudes_old = amplitudes;

  for (ulong iter = 0; iter < em_max_iter; iter++) {
    next_C.init_to_zero();
    next_mu.init_to_zero();

    parallel_run(get_n_threads(), n_nodes * n_realizations,
                 &HawkesSumGaussians::estimate_ru, this, mu, amplitudes);
    parallel_run(std::min(get_n_threads(), static_cast<const unsigned int>(n_nodes)), n_nodes,
                 &HawkesSumGaussians::update_u, this, mu, amplitudes);
  }

  parallel_run(std::min(get_n_threads(), static_cast<const unsigned int>(n_nodes)), n_nodes,
               &HawkesSumGaussians::prox_amplitudes_u, this, amplitudes, amplitudes_old);
}

// Procedure called by HawkesSumGaussians::solve
void HawkesSumGaussians::estimate_ru(const ulong r_u,
                                     ArrayDouble &mu, ArrayDouble2d &amplitudes) {
  // Obtain realization and node index from r_u
  const ulong r = static_cast<const ulong>(r_u / n_nodes);
  const ulong node_u = r_u % n_nodes;

  // Fetch corresponding data
  SArrayDoublePtrList1D &realization = timestamps_list[r];
  ArrayDouble amplitudes_u = view_row(amplitudes, node_u);
  ArrayDouble2d g_ru = view(g[r][node_u]);
  double mu_u = mu[node_u];

  // initialize next data
  double &next_mu_ur = next_mu(r, node_u);
  ArrayDouble next_C_ru = view_row(next_C, r * n_nodes + node_u);
  ArrayDouble unnormalized_next_C_ru = view_row(unnormalized_next_C, r * n_nodes + node_u);

  // We loop in reverse order to benefit from last_indices
  for (ulong i = realization[node_u]->size() - 1; i != static_cast<ulong>(-1); i--) {
    // this array will store temporary values
    unnormalized_next_C_ru.init_to_zero();
    ArrayDouble g_ru_i = view_row(g_ru, i);

    // norm will be equal to mu_u + \sum_v \sum_m a_uv^m \sum_(t_j < t_i) g_m(t_i - t_j)
    double norm = mu_u;

    for (ulong node_v = 0; node_v < n_nodes; node_v++) {
      for (ulong m = 0; m < n_gaussians; m++) {
        const double sum_unnormalized_p_ij =
          amplitudes_u[node_v * n_gaussians + m] * g_ru_i[node_v * n_gaussians + m];
        unnormalized_next_C_ru[node_v * n_gaussians + m] += sum_unnormalized_p_ij;
        norm += sum_unnormalized_p_ij;
      }
    }
    next_mu_ur += mu_u / norm;
    next_C_ru.mult_incr(unnormalized_next_C_ru, 1. / norm);
  }
}

// A method called in parallel by the method 'solve' (see below)
void HawkesSumGaussians::update_u(const ulong u, ArrayDouble &mu, ArrayDouble2d &amplitudes) {
  ArrayDouble amplitudes_u = view_row(amplitudes, u);
  update_amplitudes_u(u, amplitudes_u);
  update_baseline_u(u, mu);
}

// Procedure called by HawkesSumGaussians::solve
void HawkesSumGaussians::update_amplitudes_u(const ulong u, ArrayDouble &amplitudes_u) {
  for (ulong v = 0; v < n_nodes; v++) {
    // computation of updated values
    // computation of ||a_uv||_2
    double norm_auv_sq = 0;
    for (ulong m = 0; m < n_gaussians; m++) {
      norm_auv_sq += amplitudes_u[v * n_gaussians + m] * amplitudes_u[v * n_gaussians + m];
    }
    double norm_auv = std::sqrt(norm_auv_sq);
    // computation of A, B and C (same notation in the paper)
    double A = (norm_auv != 0. ? strength_grouplasso / norm_auv : 0.);
    for (ulong m = 0; m < n_gaussians; m++) {
      double B = kernel_integral[v * n_gaussians + m] + strength_lasso;
      double C = 0;
      for (ulong r = 0; r < n_realizations; r++) {
        C -= next_C(r * n_nodes + u, v * n_gaussians + m);
      }
      // computation of updated value
      double sol = 0.;
      if (A != 0.) {
        sol += (-B + std::sqrt(B * B - 4 * A * C)) / (2 * A);
      } else {
        sol -= C / B;
      }
      amplitudes_u[v * n_gaussians + m] = sol;
    }
  }
}

// Procedure called by HawkesSumGaussians::solve
void HawkesSumGaussians::prox_amplitudes_u(const ulong u,
                                           ArrayDouble2d &amplitudes,
                                           ArrayDouble2d &amplitudes_old) {
  ArrayDouble amplitudes_u = view_row(amplitudes, u);

  ArrayDouble amplitudes_u_old = view_row(amplitudes_old, u);
  ArrayDouble grad_Q(n_gaussians);

  for (ulong v = 0; v < n_nodes; v++) {
    // computation of updated values
    grad_Q.init_to_zero();
    for (ulong m = 0; m < n_gaussians; m++) {
      // computation of the constant C
      double C = 0;
      for (ulong r = 0; r < n_realizations; r++) {
        C -= next_C(r * n_nodes + u, v * n_gaussians + m);
      }
      // computation of the gradient at (u,v)
      grad_Q[m] +=
        (amplitudes_u_old[v * n_gaussians + m] != 0. ?
         kernel_integral[v * n_gaussians + m] + C / amplitudes_u_old[v * n_gaussians + m] :
         kernel_integral[v * n_gaussians + m]);
    }
    // check is thresholding condition is met
    double diff_norm = 0.;
    double tmp;
    for (ulong m = 0; m < n_gaussians; m++) {
      tmp = soft_thres(amplitudes_u[v * n_gaussians + m]
                         - step_size * grad_Q[m], step_size * strength_lasso);
      diff_norm += tmp * tmp;
    }
    diff_norm = std::sqrt(diff_norm);
    // if condition is met set a^{(k+1)}_{uv} to zero, else use the thresholding
    // operator (see Eq. 8 of the paper)
    if (diff_norm <= step_size * strength_grouplasso) {
      for (ulong m = 0; m < n_gaussians; m++) {
        amplitudes_u[v * n_gaussians + m] = 0.;
      }
    } else {
      for (ulong m = 0; m < n_gaussians; m++) {
        amplitudes_u[v * n_gaussians + m] =
          std::max(1. - step_size * strength_grouplasso / diff_norm, 0.) *
            soft_thres(amplitudes_u[v * n_gaussians + m] - step_size * grad_Q[m],
                       step_size * strength_lasso);
      }
    }
  }
}

void HawkesSumGaussians::update_baseline_u(const ulong u, ArrayDouble &mu) {
  mu[u] = 0;
  for (ulong r = 0; r < n_realizations; r++) {
    mu[u] += next_mu(r, u) / end_times->sum();
  }
}

ulong HawkesSumGaussians::get_n_gaussians() const {
  return n_gaussians;
}

void HawkesSumGaussians::set_n_gaussians(const ulong n_gaussians) {
  if (n_gaussians <= 0) {
    TICK_ERROR("n_gaussians must be positive, received " << n_gaussians);
  }
  this->n_gaussians = n_gaussians;
  weights_computed = false;
}

ulong HawkesSumGaussians::get_em_max_iter() const {
  return em_max_iter;
}

void HawkesSumGaussians::set_em_max_iter(const ulong em_max_iter) {
  if (em_max_iter <= 0) {
    TICK_ERROR("em_max_iter must be positive, received " << em_max_iter);
  }
  this->em_max_iter = em_max_iter;
}

double HawkesSumGaussians::get_max_mean_gaussian() const {
  return max_mean_gaussian;
}

void HawkesSumGaussians::set_max_mean_gaussian(const double max_mean_gaussian) {
  if (max_mean_gaussian <= 0) {
    TICK_ERROR("max_mean_gaussian must be positive, received " << max_mean_gaussian);
  }
  this->max_mean_gaussian = max_mean_gaussian;
  weights_computed = false;
}

double HawkesSumGaussians::get_step_size() const {
  return step_size;
}

void HawkesSumGaussians::set_step_size(const double step_size) {
  if (step_size <= 0) {
    TICK_ERROR("step_size must be positive, received " << step_size);
  }
  this->step_size = step_size;
}

double HawkesSumGaussians::get_strength_lasso() const {
  return strength_lasso;
}

void HawkesSumGaussians::set_strength_lasso(const double strength_lasso) {
  if (strength_lasso <= 0) {
    TICK_ERROR("strength_lasso must be positive, received " << strength_lasso);
  }
  this->strength_lasso = strength_lasso;
}

double HawkesSumGaussians::get_strength_grouplasso() const {
  return strength_grouplasso;
}

void HawkesSumGaussians::set_strength_grouplasso(const double strength_grouplasso) {
  if (strength_grouplasso <= 0) {
    TICK_ERROR("strength_grouplasso must be positive, received " << strength_grouplasso);
  }
  this->strength_grouplasso = strength_grouplasso;
}
