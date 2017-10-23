// License: BSD 3 clause

%module simulation

%include defs.i
%include serialization.i

%{
#include "tick_python.h"
%}
%import(module="tick.base") base_module.i

%{
#include "pp.h"
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

  bool get_threshold_negative_intensity() const;
  void set_threshold_negative_intensity(const bool threshold_negative_intensity);
};


%include poisson.i
%include inhomogeneous_poisson.i
%include hawkes.i

%include hawkes_kernels.i
