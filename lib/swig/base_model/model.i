// License: BSD 3 clause


%{
#include "tick/base_model/model.h"
%}

// We need to expose this empty class to Python, so that we can pass
// shared pointers of it to models
class Model {

 public:

  Model() { }

  virtual void grad(const ArrayDouble& coeffs, ArrayDouble& out);
  virtual double loss(const ArrayDouble& coeffs);

  virtual unsigned long get_epoch_size() const;

  virtual bool is_sparse() const;
};

typedef std::shared_ptr<Model> ModelPtr;
