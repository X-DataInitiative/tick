

%include std_shared_ptr.i
%shared_ptr(HawkesCumulant);

%{
#include "tick/hawkes/inference/hawkes_cumulant.h"
%}


class HawkesCumulant : public ModelHawkesList {

public:
  HawkesCumulant(double integration_support);

  SArrayDoublePtr compute_A_and_I_ij(ulong r, ulong i, ulong j, double mean_intensity_j);

  double compute_E_ijk(ulong r, ulong i, ulong j, ulong k,
                       double mean_intensity_i, double mean_intensity_j,
                       double J_ij);

  double get_integration_support() const;
  void set_integration_support(const double integration_support);
  bool get_are_cumulants_ready() const;
  void set_are_cumulants_ready(const bool recompute_cumulants);
};

class HawkesTheoreticalCumulant{
public:
  HawkesTheoreticalCumulant(int);
  int get_dimension();
  void set_baseline(const SArrayDoublePtr mu);
  SArrayDoublePtr get_baseline();
  void set_g_geom(const SArrayDouble2dPtr R);
  SArrayDouble2dPtr get_g_geom();
  void compute_cumulants();
  SArrayDoublePtr mean_intensity();
  SArrayDouble2dPtr covariance();
  SArrayDouble2dPtr skewness();
};
