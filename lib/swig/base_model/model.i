// License: BSD 3 clause


%{
#include "tick/base_model/model.h"
%}

// We need to expose this empty class to Python, so that we can pass
// shared pointers of it to models

template <class T>
class TModel{
 public:
  TModel(){}
  virtual void grad(const Array<T>& coeffs, Array<T>& out);
  virtual T loss(const Array<T>& coeffs);
  virtual unsigned long get_epoch_size() const;
  virtual bool is_sparse() const;  
};

%rename(Model) TModel<double>;
class TModel<double>{
 public:
  Model(){}
  virtual void grad(const ArrayDouble& coeffs, ArrayDouble& out);
  virtual double loss(const ArrayDouble& coeffs);
  virtual unsigned long get_epoch_size() const;
  virtual bool is_sparse() const;  
};
typedef TModel<double> Model;
typedef std::shared_ptr<Model> ModelPtr;

%rename(ModelDouble) TModel<double>;
class TModel<double>{
 public:
  ModelDouble(){}
  virtual void grad(const ArrayDouble& coeffs, ArrayDouble& out);
  virtual double loss(const ArrayDouble& coeffs);
  virtual unsigned long get_epoch_size() const;
  virtual bool is_sparse() const;  
};

typedef TModel<double> ModelDouble;
typedef std::shared_ptr<ModelDouble> ModelDoublePtr;

%rename(ModelFloat) TModel<float>;
class TModel<float>{
 public:
  ModelFloat(){}
  virtual void grad(const ArrayFloat& coeffs, ArrayFloat& out);
  virtual double loss(const ArrayFloat& coeffs);
  virtual unsigned long get_epoch_size() const;
  virtual bool is_sparse() const;  
};
typedef TModel<float> ModelFloat;
typedef std::shared_ptr<ModelFloat> ModelFloatPtr;
