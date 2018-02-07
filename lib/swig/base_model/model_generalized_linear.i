// License: BSD 3 clause


%{
#include "tick/base_model/model_generalized_linear.h"
%}

%include "model_labels_features.i"

template <class T>
class TModelGeneralizedLinear : public virtual TModelLabelsFeatures<T> {
 public:
  TModelGeneralizedLinear(
    const std::shared_ptr<BaseArray2d<T> > features,
    const std::shared_ptr<SArray<T> > labels,
    const bool fit_intercept,
    const int n_threads = 1
  );
  unsigned long get_n_coeffs() const override;
  virtual void set_fit_intercept(bool fit_intercept);
  void sdca_primal_dual_relation(const T l_l2sq,
                                 const Array<T> &dual_vector,
                                 Array<T> &out_primal_vector) override;
};

%rename(ModelGeneralizedLinear) TModelGeneralizedLinear<double>;
class ModelGeneralizedLinear : public virtual TModelLabelsFeatures<double>{
 public:
  ModelGeneralizedLinear(const SBaseArrayDouble2dPtr features,
                         const SArrayDoublePtr labels,
                         const bool fit_intercept,
                         const int n_threads = 1);
  unsigned long get_n_coeffs() const override;
  virtual void set_fit_intercept(bool fit_intercept);
  void sdca_primal_dual_relation(const double l_l2sq,
                                 const ArrayDouble &dual_vector,
                                 ArrayDouble &out_primal_vector);
};
typedef TModelGeneralizedLinear<double> ModelGeneralizedLinear;

%rename(ModelGeneralizedLinearDouble) TModelGeneralizedLinear<double>;
class TModelGeneralizedLinear<double> : public virtual TModelLabelsFeatures<double>{
 public:
  TModelGeneralizedLinear(const SBaseArrayDouble2dPtr features,
                         const SArrayDoublePtr labels,
                         const bool fit_intercept,
                         const int n_threads = 1);
  unsigned long get_n_coeffs() const override;
  virtual void set_fit_intercept(bool fit_intercept);
  void sdca_primal_dual_relation(const double l_l2sq,
                                 const ArrayDouble &dual_vector,
                                 ArrayDouble &out_primal_vector);
};
typedef TModelGeneralizedLinear<double> ModelGeneralizedLinearDouble;

%rename(ModelGeneralizedLinearFloat) TModelGeneralizedLinear<float>;
class TModelGeneralizedLinear<float> : public virtual TModelLabelsFeatures<float>{
 public:
  TModelGeneralizedLinear(const SBaseArrayFloat2dPtr features,
                         const SArrayFloatPtr labels,
                         const bool fit_intercept,
                         const int n_threads = 1);
  unsigned long get_n_coeffs() const override;
  virtual void set_fit_intercept(bool fit_intercept);
  void sdca_primal_dual_relation(const float l_l2sq,
                                 const ArrayFloat &dual_vector,
                                 ArrayFloat &out_primal_vector);
};
typedef TModelGeneralizedLinear<float> ModelGeneralizedLinearFloat;
