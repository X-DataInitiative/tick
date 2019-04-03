// License: BSD 3 clause

%include "std_vector.i"
%include std_shared_ptr.i

%{
#include "tick/prox/prox.h"
%}

template <class T, class K>
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
    const Array<K> &coeffs,
    T step,
    Array<K> &out
  );
  virtual T value(const Array<K> &coeffs);
  virtual T get_strength() const;
  virtual void set_strength(T strength);
  virtual ulong get_start() const;
  virtual ulong get_end() const;
  virtual void set_start_end(ulong start, ulong end);
  virtual bool get_positive() const;
  virtual void set_positive(bool positive);
};

%rename(Prox) TProx<double, double>;
class TProx<double, double> {
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
typedef TProx<double, double> Prox;

%rename(ProxDouble) TProx<double, double>;
class TProx<double, double> {
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

typedef TProx<double, double> ProxDouble;
%rename(ProxDoublePtr) std::shared_ptr<ProxDouble>;
typedef std::shared_ptr<ProxDouble> ProxDoublePtr;

%rename(ProxFloat) TProx<float, float>;
class TProx<float, float> {
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
typedef TProx<float, float> ProxFloat;
%rename(ProxFloatPtr) std::shared_ptr<ProxFloat>;
typedef std::shared_ptr<ProxFloat> ProxFloatPtr;

%rename(ProxAtomicDouble) TProx<double, std::atomic<double>>;
class TProx<double, std::atomic<double>> {
 public:
  ProxAtomicDouble(
    double strength,
    bool positive
  );
  ProxAtomicDouble(
    double strength,
    unsigned long start,
    unsigned long end,
    bool positive
  );
  virtual void call(
    const Array<std::atomic<double>> &coeffs,
    double step,
    Array<std::atomic<double>> &out
  );
  virtual double value(const ArrayAtomicDouble &coeffs);
  virtual double get_strength() const;
  virtual void set_strength(double strength);
  virtual ulong get_start() const;
  virtual ulong get_end() const;
  virtual void set_start_end(ulong start, ulong end);
  virtual bool get_positive() const;
  virtual void set_positive(bool positive);
};
typedef TProx<double, std::atomic<double> > ProxAtomicDouble;
typedef std::shared_ptr<ProxAtomicDouble> ProxAtomicDoublePtr;

%template(TProxAtomicFloat) TProx<float, std::atomic<float> >;
typedef TProx<float, std::atomic<float> > TProxAtomicFloat;
typedef std::shared_ptr<ProxAtomicFloat> ProxAtomicFloatPtr;
