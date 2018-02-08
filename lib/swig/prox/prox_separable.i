// License: BSD 3 clause

%{
#include "tick/prox/prox_separable.h"
%}

%include "prox.i"

template <class T>
class TProxSeparable : public TProx<T> {
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

  using TProx<T>::call;

  virtual void call(
    const Array<T> &coeffs,
    const Array<T> &step,
    Array<T> &out
  );
};

%rename(TProxSeparableDouble) TProxSeparable<double>;
class TProxSeparable<double> : public TProx<double> {
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

  using TProx<double>::call;

  virtual void call(
    const ArrayDouble &coeffs,
    const ArrayDouble &step,
    ArrayDouble &out
  );
};
typedef TProxSeparable<double> TProxSeparableDouble;

%rename(TProxSeparableFloat) TProxSeparable<float>;
class TProxSeparable<float> : public TProx<float> {
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

  using TProx<float>::call;

  virtual void call(
    const ArrayFloat &coeffs,
    const ArrayFloat &step,
    ArrayFloat &out
  );
};
typedef TProxSeparable<float> TProxSeparableFloat;