// License: BSD 3 clause

%module hawkes

%include base_model_module.i

%shared_ptr(ModelHawkes);

%shared_ptr(ModelHawkesSingle);
%shared_ptr(ModelHawkesFixedExpKernLogLik);
%shared_ptr(ModelHawkesFixedSumExpKernLogLik);
%shared_ptr(ModelHawkesFixedExpKernLeastSq);
%shared_ptr(ModelHawkesFixedSumExpKernLeastSq);

%shared_ptr(ModelHawkesList);
%shared_ptr(ModelHawkesLeastSqList);
%shared_ptr(ModelHawkesFixedKernLogLikList);
%shared_ptr(ModelHawkesFixedExpKernLeastSqList);
%shared_ptr(ModelHawkesFixedSumExpKernLeastSqList);
%shared_ptr(ModelHawkesFixedExpKernLogLikList);
%shared_ptr(ModelHawkesFixedSumExpKernLogLikList);

%{
#include "tick/base/tick_python.h"
%}

%import(module="tick.base") base_module.i

%include model/hawkes.i


%include defs.i
%include serialization.i

%{
#include "tick/hawkes/model/base/hawkes_list.h"
%}
%include inference/hawkes_conditional_law.i
%include inference/hawkes_em.i
%include inference/hawkes_adm4.i
%include inference/hawkes_basis_kernels.i
%include inference/hawkes_sumgaussians.i

%{
#include "tick/hawkes/simulation/pp.h"
%}

class PP {
    
 public :

  PP(unsigned int n_nodes, int seed = -1);
  virtual ~PP();

  void activate_itr(double dt);

  void simulate(double run_time);
  void simulate(ulong  n_points);
  void simulate(double run_time, ulong n_points);
  virtual void reset();
  //    bool flagThresholdNegativeIntensity;
  bool itr_on();

  double get_time();
  unsigned int get_n_nodes();
  int get_seed() const;
  ulong get_n_total_jumps();
  VArrayDoublePtrList1D get_itr();
  VArrayDoublePtr get_itr_times();
  double get_itr_step();

  void reseed_random_generator(int seed);

  SArrayDoublePtrList1D get_timestamps();
  void set_timestamps(VArrayDoublePtrList1D &timestamps, double end_time);

  bool get_threshold_negative_intensity() const;
  void set_threshold_negative_intensity(const bool threshold_negative_intensity);
};


%include simulation/poisson.i
%include simulation/inhomogeneous_poisson.i
%include simulation/hawkes.i
%include simulation/hawkes_kernels.i
