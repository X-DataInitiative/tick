// License: BSD 3 clause


%{
#include "tick/base_model/model.h"
%}

// We need to expose this empty class to Python, so that we can pass
// shared pointers of it to models

template <class T, class K = T>
class TModel{
 public:
  TModel(){}
  virtual void grad(const Array<T>& coeffs, Array<T>& out);
  virtual T loss(const Array<T>& coeffs);
  virtual unsigned long get_epoch_size() const;
  virtual bool is_sparse() const;
};

%rename(Model) TModel<double, double>;
class TModel< double,double > {
 public:
  ModelDouble(){}
  virtual void grad(const ArrayDouble& coeffs, ArrayDouble& out);
  virtual double loss(const ArrayDouble& coeffs);
  virtual unsigned long get_epoch_size() const;
  virtual bool is_sparse() const;
};
typedef TModel<double, double> Model;
typedef std::shared_ptr<Model> ModelPtr;

%rename(ModelDouble) TModel<double, double>;
class TModel< double,double > {
 public:
  ModelDouble(){}
  virtual void grad(const ArrayDouble& coeffs, ArrayDouble& out);
  virtual double loss(const ArrayDouble& coeffs);
  virtual unsigned long get_epoch_size() const;
  virtual bool is_sparse() const;
};
typedef TModel<double, double> ModelDouble;
typedef std::shared_ptr<ModelDouble> ModelDoublePtr;

%rename(ModelFloat) TModel<float, float>;
class TModel<float, float> {
 public:
  ModelFloat(){}
  virtual void grad(const ArrayFloat& coeffs, ArrayFloat& out);
  virtual double loss(const ArrayFloat& coeffs);
  virtual unsigned long get_epoch_size() const;
  virtual bool is_sparse() const;
};
typedef TModel<float, float> ModelFloat;
typedef std::shared_ptr<ModelFloat> ModelFloatPtr;


%rename(ModelAtomicDouble) TModel<double, std::atomic<double> >;
class TModel<double, std::atomic<double> >{
 public:
  ModelAtomicDouble(){}
  virtual void grad(const ArrayAtomicDouble& coeffs, ArrayDouble& out);
  virtual double loss(const ArrayAtomicDouble& coeffs);
  virtual unsigned long get_epoch_size() const;
  virtual bool is_sparse() const;
};

typedef TModel<double, std::atomic<double> > ModelAtomicDouble;
typedef std::shared_ptr<ModelAtomicDouble> ModelAtomicDoublePtr;


%rename(ModelAtomicFloat) TModel<float, std::atomic<float> >;
class TModel<float, std::atomic<float> >{
 public:
  ModelAtomicFloat(){}
  virtual void grad(const Array<std::atomic<float>>& coeffs, ArrayFloat& out);
  virtual float loss(const Array<std::atomic<float>>& coeffs);
  virtual unsigned long get_epoch_size() const;
  virtual bool is_sparse() const;
};
typedef TModel<float, std::atomic<float> > ModelAtomicFloat;

