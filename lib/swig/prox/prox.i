// License: BSD 3 clause

%include "std_vector.i"
%include std_shared_ptr.i

%{
#include "tick/prox/prox.h"
%}

template <class T>
class TProx {
 public:
  TProx(
    T strength,
    bool positive
  );
  TProx(
    T strength,
    unsigned long start,
    unsigned long end,
    bool positive
  );
  virtual void call(
    const Array<T> &coeffs,
    T step,
    Array<T> &out
  );
  virtual T value(const Array<T> &coeffs);
  virtual T get_strength() const;
  virtual void set_strength(T strength);
  virtual ulong get_start() const;
  virtual ulong get_end() const;
  virtual void set_start_end(ulong start, ulong end);
  virtual bool get_positive() const;
  virtual void set_positive(bool positive);
};

%rename(Prox) TProx<double>;
class TProx<double> {
 public:
  Prox(
    double strength,
    bool positive
  );
  Prox(
    double strength,
    unsigned long start,
    unsigned long end,
    bool positive
  );
  virtual void call(
    const ArrayDouble &coeffs,
    double step,
    ArrayDouble &out
  );
  virtual double value(const ArrayDouble &coeffs);
  virtual double get_strength() const;
  virtual void set_strength(double strength);
  virtual ulong get_start() const;
  virtual ulong get_end() const;
  virtual void set_start_end(ulong start, ulong end);
  virtual bool get_positive() const;
  virtual void set_positive(bool positive);
};
typedef TProx<double> Prox;

%rename(ProxDouble) TProx<double>;
class TProx<double> {
 public:
  ProxDouble(
    double strength,
    bool positive
  );
  ProxDouble(
    double strength,
    unsigned long start,
    unsigned long end,
    bool positive
  );
  virtual void call(
    const ArrayDouble &coeffs,
    double step,
    ArrayDouble &out
  );
  virtual double value(const ArrayDouble &coeffs);
  virtual double get_strength() const;
  virtual void set_strength(double strength);
  virtual ulong get_start() const;
  virtual ulong get_end() const;
  virtual void set_start_end(ulong start, ulong end);
  virtual bool get_positive() const;
  virtual void set_positive(bool positive);
};
typedef TProx<double> ProxDouble;

%rename(ProxDoublePtr) std::shared_ptr<ProxDouble>;
typedef std::shared_ptr<ProxDouble> ProxDoublePtr;

%rename(ProxFloat) TProx<float>;
class TProx<float> {
 public:
  ProxFloat(
    float strength,
    bool positive
  );
  ProxFloat(
    float strength,
    unsigned long start,
    unsigned long end,
    bool positive
  );
  virtual void call(
    const ArrayFloat &coeffs,
    float step,
    ArrayFloat &out
  );
  virtual float value(const ArrayFloat &coeffs);
  virtual float get_strength() const;
  virtual void set_strength(float strength);
  virtual ulong get_start() const;
  virtual ulong get_end() const;
  virtual void set_start_end(ulong start, ulong end);
  virtual bool get_positive() const;
  virtual void set_positive(bool positive);
};
typedef TProx<float> ProxFloat;
typedef std::shared_ptr<ProxFloat> ProxFloatPtr;
