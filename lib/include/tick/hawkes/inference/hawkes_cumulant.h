#ifndef LIB_INCLUDE_TICK_HAWKES_INFERENCE_HAWKES_CUMULANT_H_
#define LIB_INCLUDE_TICK_HAWKES_INFERENCE_HAWKES_CUMULANT_H_

// License: BSD 3 clause

#include "tick/base/base.h"
#include "tick/hawkes/model/base/model_hawkes_list.h"

class DLL_PUBLIC HawkesCumulant : public ModelHawkesList {
  double integration_support;
  bool are_cumulants_ready;

 public:
  explicit HawkesCumulant(double integration_support);

  SArrayDoublePtr compute_A_and_I_ij(ulong r, ulong i, ulong j, double mean_intensity_j);

  double compute_E_ijk(ulong r, ulong i, ulong j, ulong k, double mean_intensity_i,
                       double mean_intensity_j, double J_ij);

  double get_integration_support() const { return integration_support; }

  void set_integration_support(const double integration_support) {
    if (integration_support <= 0) TICK_ERROR("Kernel support must be positive");
    this->integration_support = integration_support;
    are_cumulants_ready = false;
  }

  bool get_are_cumulants_ready() const { return are_cumulants_ready; }

  void set_are_cumulants_ready(const bool are_cumulants_ready) {
    this->are_cumulants_ready = are_cumulants_ready;
  }
};

class DLL_PUBLIC HawkesTheoreticalCumulant {
 private:
  int d;
  SArrayDoublePtr mu;
  SArrayDoublePtr first_cumulant;     // The matrix \$Lambda\$ from the paper
  SArrayDouble2dPtr second_cumulant;  // The matrix \$C\$ from the paper
  SArrayDouble2dPtr third_cumulant;   // The matrix \$Kc\$ from the paper
  SArrayDouble2dPtr g_geom;           // The matrix R = (I - G)^{-1} from the paper

 public:
  HawkesTheoreticalCumulant(int);
  int get_dimension() { return d; }
  void set_baseline(SArrayDoublePtr const mu) { this->mu = mu; }
  SArrayDoublePtr get_baseline() { return mu; }
  void set_g_geom(SArrayDouble2dPtr const g_geom) { this->g_geom = g_geom; }
  SArrayDouble2dPtr get_g_geom() { return g_geom; }
  void compute_mean_intensity();
  void compute_covariance();
  void compute_skewness();
  void compute_cumulants();
  SArrayDoublePtr mean_intensity() { return first_cumulant; }
  SArrayDouble2dPtr covariance() { return second_cumulant; }
  SArrayDouble2dPtr skewness() { return third_cumulant; }
};

#endif  // LIB_INCLUDE_TICK_HAWKES_INFERENCE_HAWKES_CUMULANT_H_
