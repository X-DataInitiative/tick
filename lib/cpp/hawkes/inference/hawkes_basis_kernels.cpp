// License: BSD 3 clause


#include "tick/hawkes/inference/hawkes_basis_kernels.h"

// Procedure called by HawkesBasisKernels::solve
// Not commented, see LaTeX notes
void compute_r(ArrayDouble &u_realization,
               double T,
               double kernel_dt,
               ArrayDouble2d &gdm,
               ArrayDouble2d &Gdm,
               ArrayDouble &rd) {
  ulong D = gdm.n_rows();
  ulong M = gdm.n_cols();
  for (ulong j = 0; j < u_realization.size(); j++) {
    ulong m0 = static_cast<ulong>(std::floor((T - u_realization[j]) / kernel_dt));
    for (ulong d = 0; d < D; d++) {
      if (m0 >= M) {
        rd[d] += Gdm[d * M + M - 1];
      } else if (m0 > 0) {
        rd[d] += Gdm[d * M + m0 - 1] + (T - u_realization[j] - m0 * kernel_dt) * gdm[d * M + m0];
      } else {
        rd[d] += (T - u_realization[j] - m0 * kernel_dt) * gdm[d * M + m0];
      }
    }
  }
}

// Procedure called by HawkesBasisKernels::solve
// Not commented, see LaTeX notes
void compute_C(ArrayDouble &u_realization,
               double T,
               double kernel_dt,
               ArrayDouble2d &gdm,
               ArrayDouble &a_sum,
               ArrayDouble2d &Cdm) {
  ulong D = gdm.n_rows();
  ulong M = gdm.n_cols();

  ulong i = u_realization.size() - 1;
  for (ulong m = 0; m < M; m++) {
    while (i != static_cast<ulong>(-1) && u_realization[i] > T - m * kernel_dt) i--;
    if (i == static_cast<ulong>(-1)) break;
    for (ulong d = 0; d < D; d++) {
      Cdm[d * M + m] += a_sum[d] * (i + 1) / gdm[d * M + m];
    }
  }
}

// Procedure called by HawkesBasisKernels::solve
// Not commented, see LaTeX notes
double compute_mu_q_D(ulong u_index,
                      SArrayDoublePtrList1D &realization,
                      double T,
                      double kernel_dt,
                      ArrayDouble2d &gdm,
                      ArrayDouble2d &avd,
                      double mu,
                      ArrayDouble2d &qvd,
                      ArrayDouble2d &qvd_temp,
                      ArrayDouble2d &Ddm,
                      ArrayDouble2d &Ddm_temp) {
  ulong dim = qvd.n_rows();
  ulong M = gdm.n_cols();
  ulong D = gdm.n_rows();

  ArrayDouble &u = *realization[u_index];

  ArrayULong v_indices(dim);
  for (ulong n = 0; n < dim; n++) {
    v_indices[n] = realization[n]->size();
  }

  double mu_out = 0;

  for (ulong i = u.size() - 1; i != static_cast<ulong>(-1); i--) {
    double norm = 0;
    qvd_temp.init_to_zero();
    Ddm_temp.init_to_zero();

    double t_i = u[i];

    for (ulong v_index = 0; v_index < dim; v_index++) {
      ArrayDouble &v = *realization[v_index];

      while (true) {
        if (v_indices[v_index] == 0) break;
        if (v_indices[v_index] < v.size() && t_i >= v[v_indices[v_index]]) break;
        v_indices[v_index]--;
      }
      if (t_i < v[v_indices[v_index]]) continue;

      ulong j0 = v_indices[v_index];

      ArrayDouble ad = view_row(avd, v_index);
      ArrayDouble qd_temp = view_row(qvd_temp, v_index);

      for (ulong j = j0; j != static_cast<ulong>(-1); j--) {
        double t_j = v[j];
        if (u_index == v_index && i == j) {
          norm += mu;
        } else {
          ulong m = static_cast<ulong>(std::floor((t_i - t_j) / kernel_dt));
          if (m >= M) break;
          for (ulong d = 0; d < D; d++) {
            double val = ad[d] * gdm[d * M + m];
            qd_temp[d] += val;
            Ddm_temp[d * M + m] += val;
            norm += val;
          }
        }
      }
    }

    mu_out += mu / norm;
    for (ulong d = 0; d < D; d++) {
      for (ulong m = 0; m < M; m++) {
        Ddm[d * M + m] += Ddm_temp[d * M + m] / (norm * kernel_dt);
      }
      for (ulong v_index = 0; v_index < dim; v_index++) {
        qvd[v_index * D + d] += avd[v_index * D + d] * qvd_temp[v_index * D + d] / norm;
      }
    }
  }

  return mu_out;
}

// Procedure called by HawkesBasisKernels::solve
// Not commented, see LaTeX notes
double compute_gdm(double alpha,
                   double kernel_dx,
                   ArrayDouble &gdm,
                   ArrayDouble &Cdm,
                   ArrayDouble &Ddm,
                   double tol,
                   ulong max_iter) {
  gdm.init_to_zero();
  ulong M = gdm.size();

  double max_rel_error = 0;

  for (ulong n_iter = 0; n_iter < max_iter; ++n_iter) {
    max_rel_error = -1;

    for (ulong m = 0; m < M; m++) {
      const double mm = m == 0 ? 0 : gdm[m - 1];
      const double pm = (m == gdm.size() - 1) ? 0 : gdm[m + 1];

      double a = 4 * alpha / (kernel_dx * kernel_dx) + Cdm[m];
      double b = -2 * alpha * (pm + mm) / (kernel_dx * kernel_dx);
      double c = -Ddm[m];

      double sol = (-b + sqrt(b * b - 4 * a * c)) / (2 * a);
      if (n_iter != 0) {
        double rel_error = gdm[m] == 0 ? sol : (sol - gdm[m]) / gdm[m];
        max_rel_error = std::max(rel_error, max_rel_error);
      }
      gdm[m] = sol;
    }

    if (n_iter > 0 && max_rel_error < tol) break;
  }

  return max_rel_error;
}

// Constructor for the main class
HawkesBasisKernels::HawkesBasisKernels(const double kernel_support, const ulong kernel_size,
                                       const ulong n_basis, const double alpha,
                                       const int max_n_threads) :
  ModelHawkesList(max_n_threads, 0) {
  set_kernel_support(kernel_support);
  set_kernel_size(kernel_size);
  set_n_basis(n_basis);
  set_alpha(alpha);
}

void HawkesBasisKernels::allocate_weights() {
  const ulong n_basis = get_n_basis();
  rud = ArrayDouble2d(n_nodes, n_basis);
  Dudm = ArrayDouble2d(n_nodes, n_basis * kernel_size);
  Gdm = ArrayDouble2d(n_basis, kernel_size);
  Dudm_temp = ArrayDouble2d(n_nodes, n_basis * kernel_size);
  Cudm = ArrayDouble2d(n_nodes, n_basis * kernel_size);
  a_sum_vd = ArrayDouble2d(n_nodes, n_basis);
  quvd_temp = ArrayDouble2d(n_nodes, n_nodes * n_basis);
  quvd = ArrayDouble2d(n_nodes, n_nodes * n_basis);

  weights_computed = true;
}

// A method called in parallel by the method 'solve' (see below)
void HawkesBasisKernels::solve_u(ulong u,
                                 ArrayDouble &mu,
                                 ArrayDouble2d &gdm,
                                 ArrayDouble2d &auvd) {
  const ulong n_basis = get_n_basis();

  ArrayDouble rd = view_row(rud, u);
  double mu_out = 0;
  ArrayDouble2d Cdm(n_basis, kernel_size, view_row(Cudm, u).data());
  ArrayDouble2d Ddm(n_basis, kernel_size, view_row(Dudm, u).data());
  ArrayDouble2d Ddm_temp(n_basis, kernel_size, view_row(Dudm_temp, u).data());
  ArrayDouble2d avd(n_nodes, n_basis, view_row(auvd, u).data());
  ArrayDouble2d qvd(n_nodes, n_basis, view_row(quvd, u).data());
  ArrayDouble2d qvd_temp(n_nodes, n_basis, view_row(quvd_temp, u).data());
  ArrayDouble a_sum_v = view_row(a_sum_vd, u);

  for (ulong r = 0; r < n_realizations; r++) {
    compute_r(*(timestamps_list[r][u]), (*end_times)[r], get_kernel_dt(), gdm, Gdm, rd);
    compute_C(*timestamps_list[r][u], (*end_times)[r], get_kernel_dt(), gdm, a_sum_v, Cdm);
    mu_out += compute_mu_q_D(u, timestamps_list[r], (*end_times)[r], get_kernel_dt(), gdm,
                             avd, mu[u], qvd, qvd_temp, Ddm, Ddm_temp);
  }

  mu_out /= end_times->sum();
  mu[u] = mu_out;
}

// The main method for performing one iteration
double HawkesBasisKernels::solve(ArrayDouble &mu,
                                 ArrayDouble2d &gdm,
                                 ArrayDouble2d &auvd,
                                 ulong max_iter_gdm,
                                 double max_tol_gdm) {
  if (!weights_computed) allocate_weights();

  if (mu.size() != n_nodes) {
    TICK_ERROR("baseline / mu argument must be an array of size " << n_nodes);
  }
  if (gdm.n_rows() != n_basis || gdm.n_cols() != kernel_size) {
    TICK_ERROR("basis functions / gdm argument must be an array of shape ("
                 << n_nodes << ", " << kernel_size << ")");
  }
  if (auvd.n_rows() != n_nodes || auvd.n_cols() != n_nodes * n_basis) {
    TICK_ERROR("amplitudes / auvd argument must be an array of shape ("
                 << n_nodes << ", " << n_nodes * n_basis << ")");
  }

  const ulong n_basis = get_n_basis();

  for (ulong d = 0; d < n_basis; d++) {
    Gdm[d * kernel_size] = gdm[d * kernel_size] * get_kernel_dt();
    for (ulong m = 1; m < kernel_size; m++)
      Gdm[d * kernel_size + m] =
        Gdm[d * kernel_size + m - 1] + gdm[d * kernel_size + m] * get_kernel_dt();
  }

  a_sum_vd.init_to_zero();
  for (ulong v = 0; v < n_nodes; v++) {
    for (ulong d = 0; d < n_basis; d++) {
      for (ulong u = 0; u < n_nodes; u++) {
        a_sum_vd[v * n_basis + d] += auvd[u * n_nodes * n_basis + v * n_basis + d];
      }
    }
  }

  rud.init_to_zero();
  quvd.init_to_zero();
  Dudm.init_to_zero();
  Cudm.init_to_zero();

  // Parallel loop on u to run compute_r, compute_C, Scompute_mu_q_D
  parallel_run(get_n_threads(), n_nodes, &HawkesBasisKernels::solve_u, this, mu, gdm, auvd);

  // Then we reduce the computations of Cudm and Dudm
  // We store in component u=0 the sum_u=0^n_nodes Cudm (same for Dudm)
  for (ulong u = 1; u < n_nodes; u++) {
    for (ulong d = 0; d < n_basis; d++) {
      for (ulong m = 0; m < kernel_size; m++) {
        Cudm[d * kernel_size + m] += Cudm[u * n_basis * kernel_size + d * kernel_size + m];
        Dudm[d * kernel_size + m] += Dudm[u * n_basis * kernel_size + d * kernel_size + m];
      }
    }
  }

  for (ulong u = 0; u < n_nodes; u++) {
    for (ulong v = 0; v < n_nodes; v++) {
      for (ulong d = 0; d < n_basis; d++) {
        auvd[u * n_nodes * n_basis + v * n_basis + d] =
          sqrt(
            quvd[u * n_nodes * n_basis + v * n_basis + d] / (rud[v * n_basis + d] + 2 * alpha));
      }
    }
  }

  double rerr_gdm = 0;
  for (ulong d = 0; d < n_basis; d++) {
    ArrayDouble gm = view_row(gdm, d);
    ArrayDouble2d Cdm(n_basis, kernel_size, view_row(Cudm, 0).data());
    ArrayDouble2d Ddm(n_basis, kernel_size, view_row(Dudm, 0).data());
    ArrayDouble Cm = view_row(Cdm, d);
    ArrayDouble Dm = view_row(Ddm, d);
    double rerr = compute_gdm(alpha, get_kernel_dt(), gm, Cm, Dm, max_tol_gdm, max_iter_gdm);
    rerr_gdm = std::max(rerr, rerr_gdm);
  }

  return rerr_gdm;
}

unsigned int HawkesBasisKernels::get_n_threads() const {
  return std::min(this->max_n_threads, static_cast<unsigned int>(n_nodes));
}

void HawkesBasisKernels::set_kernel_support(const double kernel_support) {
  if (kernel_support <= 0) {
    TICK_ERROR("Kernel support must be positive and you have provided " << kernel_support)
  }
  this->kernel_support = kernel_support;
}

void HawkesBasisKernels::set_kernel_size(const ulong kernel_size) {
  if (kernel_size <= 0) {
    TICK_ERROR("Kernel size must be positive and you have provided " << kernel_size)
  }
  this->kernel_size = kernel_size;
  weights_computed = false;
}

void HawkesBasisKernels::set_kernel_dt(const double kernel_dt) {
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

void HawkesBasisKernels::set_n_basis(const ulong n_basis) {
  this->n_basis = n_basis;
  weights_computed = false;
}

void HawkesBasisKernels::set_alpha(const double alpha) {
  if (alpha <= 0) {
    TICK_ERROR("alpha must be positive and you have provided " << alpha)
  }
  this->alpha = alpha;
}
