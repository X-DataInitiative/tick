// License: BSD 3 clause

%{
#include "tick/prox/prox_separable.h"
%}

%include "prox.i"

template <class T, class K>
class TProxSeparable : public TProx<T, K> {
 public:
  TProxSeparable(
    T strength,
    bool positive
  );
  TProxSeparable(
    T strength,
    unsigned long start,
    unsigned long end,
    bool positive
  );

  using TProx<T, K>::call;

  virtual void call(
    const Array<K> &coeffs,
    const Array<T> &step,
    Array<K> &out
  );
};

%rename(TProxSeparableDouble) TProxSeparable<double, double>;
class TProxSeparable<double, double> : public TProx<double, double> {
 public:
  TProxSeparableDouble(
    double strength,
    bool positive
  );
  TProxSeparableDouble(
    double strength,
    unsigned long start,
    unsigned long end,
    bool positive
  );

  using TProx<double, double>::call;

  virtual void call(
    const ArrayDouble &coeffs,
    const ArrayDouble &step,
    ArrayDouble &out
  );
};
typedef TProxSeparable<double, double> TProxSeparableDouble;

%rename(TProxSeparableFloat) TProxSeparable<float, float>;
class TProxSeparable<float, float> : public TProx<float, float> {
 public:
  TProxSeparable(
    float strength,
    bool positive
  );
  TProxSeparable(
    float strength,
    unsigned long start,
    unsigned long end,
    bool positive
  );

  using TProx<float, float>::call;

  virtual void call(
    const ArrayFloat &coeffs,
    const ArrayFloat &step,
    ArrayFloat &out
  );
};
typedef TProxSeparable<float, float> TProxSeparableFloat;

%template(TProxSeparableAtomicDouble) TProxSeparable<double, std::atomic<double> >;
typedef TProxSeparable<double, std::atomic<double> > TProxSeparableAtomicDouble;


%template(TProxSeparableAtomicFloat) TProxSeparable<float, std::atomic<float> >;
typedef TProxSeparable<float, std::atomic<double> > TProxSeparableAtomicFloat;
