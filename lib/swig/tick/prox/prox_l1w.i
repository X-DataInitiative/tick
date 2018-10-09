// License: BSD 3 clause

%{
#include "tick/prox/prox_l1w.h"
%}

template <class T, class K>
class TProxL1w : public TProxSeparable<T, K> {
 public:
  TProxL1w(T strength, std::shared_ptr<SArray<T>> weights, bool positive);

  TProxL1w(T strength, std::shared_ptr<SArray<T>> weights, ulong start, ulong end, bool positive);

  virtual void set_weights(std::shared_ptr<SArray<T>> weights) final;

  bool compare(const TProxL1w<T, K> &that);
};

%rename(ProxL1wDouble) TProxL1w<double, double>;
class TProxL1w<double, double> : public TProxSeparable<double, double> {
 public:
  ProxL1wDouble(double strength, SArrayDoublePtr weights, bool positive);

  ProxL1wDouble(double strength, SArrayDoublePtr weights, ulong start, ulong end, bool positive);

  virtual void set_weights(SArrayDoublePtr weights) final;

  bool compare(const ProxL1wDouble &that);
};
typedef TProxL1w<double, double> ProxL1wDouble;

%rename(ProxL1wFloat) TProxL1w<float, float>;
class TProxL1w<float, float> : public TProxSeparable<float, float> {
 public:
  ProxL1wFloat(float strength, SArrayFloatPtr weights, bool positive);

  ProxL1wFloat(float strength, SArrayFloatPtr weights, ulong start, ulong end, bool positive);

  virtual void set_weights(SArrayFloatPtr weights) final;

  bool compare(const ProxL1wFloat &that);
};
typedef TProxL1w<float, float> ProxL1wFloat;

