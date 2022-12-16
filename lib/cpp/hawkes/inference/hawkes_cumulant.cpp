#include "tick/hawkes/inference/hawkes_cumulant.h"

HawkesCumulant::HawkesCumulant(double integration_support)
    : integration_support(integration_support), are_cumulants_ready(false) {}

SArrayDoublePtr HawkesCumulant::compute_A_and_I_ij(ulong r, ulong i, ulong j,
                                                   double mean_intensity_j) {
  auto timestamps_i = timestamps_list[r][i];
  auto timestamps_j = timestamps_list[r][j];

  ulong n_i = timestamps_i->size();
  ulong n_j = timestamps_j->size();
  double res_C = 0;
  double res_J = 0;
  double width = 2 * integration_support;
  double trend_C_j = mean_intensity_j * width;
  double trend_J_j = mean_intensity_j * width * width;

  ulong last_l = 0;
  for (ulong k = 0; k < n_i; ++k) {
    double t_i_k = (*timestamps_i)[k];
    double t_i_k_minus_half_width = t_i_k - integration_support;
    double t_i_k_minus_width = t_i_k - width;

    if (t_i_k_minus_half_width < 0) continue;

    // Find next t_j_l that occurs width before t_i_k
    while (last_l < n_j) {
      if ((*timestamps_j)[last_l] <= t_i_k_minus_width)
        ++last_l;
      else
        break;
    }

    ulong l = last_l;
    ulong timestamps_in_interval = 0;

    double sub_res = 0.;

    while (l < n_j) {
      double t_j_l_minus_t_i_k = (*timestamps_j)[l] - t_i_k;
      double abs_t_j_l_minus_t_i_k = fabs(t_j_l_minus_t_i_k);

      if (abs_t_j_l_minus_t_i_k < width) {
        sub_res += width - abs_t_j_l_minus_t_i_k;

        if (abs_t_j_l_minus_t_i_k < integration_support) timestamps_in_interval++;
      } else {
        break;
      }
      l += 1;
    }

    if (l == n_j) continue;
    res_C += timestamps_in_interval - trend_C_j;
    res_J += sub_res - trend_J_j;
  }

  res_C /= (*end_times)[r];
  res_J /= (*end_times)[r];

  ArrayDouble return_array{res_C, res_J};
  return return_array.as_sarray_ptr();
}

double HawkesCumulant::compute_E_ijk(ulong r, ulong i, ulong j, ulong k, double mean_intensity_i,
                                     double mean_intensity_j, double J_ij) {
  auto timestamps_i = timestamps_list[r][i];
  auto timestamps_j = timestamps_list[r][j];
  auto timestamps_k = timestamps_list[r][k];

  double L_i = mean_intensity_i;
  double L_j = mean_intensity_j;

  double res = 0;
  ulong last_l = 0;
  ulong last_m = 0;
  ulong n_i = timestamps_i->size();
  ulong n_j = timestamps_j->size();
  ulong n_k = timestamps_k->size();

  double trend_i = L_i * 2 * integration_support;
  double trend_j = L_j * 2 * integration_support;

  for (ulong t = 0; t < n_k; ++t) {
    double tau = (*timestamps_k)[t];

    if (tau - integration_support < 0) continue;

    while (last_l < n_i) {
      if ((*timestamps_i)[last_l] <= tau - integration_support)
        last_l += 1;
      else
        break;
    }
    ulong l = last_l;

    while (l < n_i) {
      if ((*timestamps_i)[l] < tau + integration_support)
        l += 1;
      else
        break;
    }

    while (last_m < n_j) {
      if ((*timestamps_j)[last_m] <= tau - integration_support)
        last_m += 1;
      else
        break;
    }
    ulong m = last_m;

    while (m < n_j) {
      if ((*timestamps_j)[m] < tau + integration_support)
        m += 1;
      else
        break;
    }

    if ((m == n_j) || (l == n_i)) continue;

    res += (l - last_l - trend_i) * (m - last_m - trend_j) - J_ij;
  }
  res /= (*end_times)[r];
  return res;
}

HawkesTheoreticalCumulant::HawkesTheoreticalCumulant(int dim) : d(dim) {
  Lambda = SArrayDouble::new_ptr(dim);
  C = SArrayDouble2d::new_ptr(dim, dim);
  Kc = SArrayDouble2d::new_ptr(dim, dim);
  R = SArrayDouble2d::new_ptr(dim, dim);
}

// Formulae from eq. 7 in the paper
void HawkesTheoreticalCumulant::compute_mean_intensity() {
  for (int i = 0; i < d; ++i) {
    double lambda_i = 0;
    for (int m = 0; m < d; ++m) {
      int _im = i * d + m;
      double mu_m = (*mu)[m];
      double R_im = (*R)[_im];
      lambda_i += R_im * mu_m;
    }
    (*Lambda)[i] = lambda_i;
  }
};

// Formulae from eq. 8 in the paper
void HawkesTheoreticalCumulant::compute_covariance() {
  for (int i = 0; i < d; ++i) {
    for (int j = 0; j < d; ++j) {
      int _ij = i * d + j;
      double C_ij = 0;
      for (int m = 0; m < d; ++m) {
        int _im = i * d + m;
        int _jm = j * d + m;
        C_ij += (*Lambda)[m] * (*R)[_im] * (*R)[_jm];
      }
      (*C)[_ij] = C_ij;
    }
  }
};

// Formulae from eq. 9 in the paper
void HawkesTheoreticalCumulant::compute_skewness() {
  for (int i = 0; i < d; ++i) {
    for (int k = 0; k < d; ++k) {
      int _ik = i * d + k;
      double Kc_ik = 0;
      for (int m = 0; m < d; ++m) {
        int _im = i * d + m;
        int _km = k * d + m;
        double R_im = (*R)[_im];
        double R_km = (*R)[_km];
        double C_km = (*C)[_km];
        double C_im = (*C)[_im];
        double Lambda_m = (*Lambda)[m];
        Kc_ik += (R_im * R_im * C_km + 2 * R_im * R_km * C_im - 2 * Lambda_m * R_im * R_im * R_km);
      }
      (*Kc)[_ik] = Kc_ik;
    }
  }
};
void HawkesTheoreticalCumulant::compute_cumulants() {
  compute_mean_intensity();
  compute_covariance();
  compute_skewness();
}
