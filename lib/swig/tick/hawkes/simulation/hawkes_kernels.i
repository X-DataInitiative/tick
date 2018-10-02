// License: BSD 3 clause

%include <std_shared_ptr.i>
%shared_ptr(HawkesKernel);
%shared_ptr(HawkesKernelExp);
%shared_ptr(HawkesKernelSumExp);
%shared_ptr(HawkesKernelPowerLaw);
%shared_ptr(HawkesKernelTimeFunc);
%shared_ptr(HawkesKernel0);

class HawkesKernel {
 public:
  HawkesKernel(double support = 0);
  bool is_zero();

  double get_support();
  double get_plot_support();

  double get_value(double x);
  SArrayDoublePtr get_values(const ArrayDouble &t_values);
  virtual double get_norm(int nsteps = 10000);
};


class HawkesKernelExp : public HawkesKernel {
 public:

  static void set_fast_exp(bool flag);
  static bool get_fast_exp();

  HawkesKernelExp(double intensity, double decay);

  double get_intensity();
  double get_decay();
};

TICK_MAKE_PICKLABLE(HawkesKernelExp, 0.0, 0.0);

class HawkesKernelSumExp : public HawkesKernel {
public:

  static void set_fast_exp(bool flag);
  static bool get_fast_exp();

  HawkesKernelSumExp(const ArrayDouble &intensities, const ArrayDouble &decays);
  HawkesKernelSumExp();

  SArrayDoublePtr get_intensities();
  SArrayDoublePtr get_decays();
  ulong get_n_decays() { return n_decays; }
};

TICK_MAKE_PICKLABLE(HawkesKernelSumExp);


class HawkesKernelPowerLaw : public HawkesKernel {
 public :

  HawkesKernelPowerLaw(double multiplier,
                       double cutoff,
                       double exponent,
                       double support = -1,
                       double error = 1e-5);

  double get_multiplier();
  double get_exponent();
  double get_cutoff();
};

TICK_MAKE_PICKLABLE(HawkesKernelPowerLaw, 0.0, 1.0, 1.0);


class HawkesKernelTimeFunc : public HawkesKernel {
 public :

  HawkesKernelTimeFunc(TimeFunction time_function);
  HawkesKernelTimeFunc(const ArrayDouble &t_axis, const ArrayDouble &y_axis);
  HawkesKernelTimeFunc();
  TimeFunction get_time_function();
};

TICK_MAKE_PICKLABLE(HawkesKernelTimeFunc);


class HawkesKernel0 : public HawkesKernel {
 public :

  HawkesKernel0();
};

TICK_MAKE_PICKLABLE(HawkesKernel0);
