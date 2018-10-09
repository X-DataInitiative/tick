// License: BSD 3 clause


%{
#include "tick/base_model/model_generalized_linear.h"
%}

%include "model_labels_features.i"

template <class T, class K>
class TModelGeneralizedLinear : public virtual TModelLabelsFeatures<T, K> {
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

%rename(ModelGeneralizedLinearDouble) TModelGeneralizedLinear<double, double>;
class TModelGeneralizedLinear<double, double> : public virtual TModelLabelsFeatures<double, double>{
 public:
  ModelGeneralizedLinearDouble(const SBaseArrayDouble2dPtr features,
                         const SArrayDoublePtr labels,
                         const bool fit_intercept,
                         const int n_threads = 1);
  unsigned long get_n_coeffs() const override;
  virtual void set_fit_intercept(bool fit_intercept);
  void sdca_primal_dual_relation(const double l_l2sq,
                                 const ArrayDouble &dual_vector,
                                 ArrayDouble &out_primal_vector);
};
typedef TModelGeneralizedLinear<double, double> ModelGeneralizedLinearDouble;

%rename(ModelGeneralizedLinearFloat) TModelGeneralizedLinear<float, float>;
class TModelGeneralizedLinear<float, float> : public virtual TModelLabelsFeatures<float, float>{
 public:
  ModelGeneralizedLinearFloat(const SBaseArrayFloat2dPtr features,
                         const SArrayFloatPtr labels,
                         const bool fit_intercept,
                         const int n_threads = 1);
  unsigned long get_n_coeffs() const override;
  virtual void set_fit_intercept(bool fit_intercept);
  void sdca_primal_dual_relation(const float l_l2sq,
                                 const ArrayFloat &dual_vector,
                                 ArrayFloat &out_primal_vector);
};
typedef TModelGeneralizedLinear<float, float> ModelGeneralizedLinearFloat;


%rename(ModelGeneralizedLinearAtomicDouble) TModelGeneralizedLinear<double, std::atomic<double> >;
class TModelGeneralizedLinear<double, std::atomic<double> > : public virtual TModelLabelsFeatures<double, std::atomic<double> >{
 public:
  ModelGeneralizedLinearAtomicDouble(const SBaseArrayDouble2dPtr features,
                         const SArrayDoublePtr labels,
                         const bool fit_intercept,
                         const int n_threads = 1);
  unsigned long get_n_coeffs() const override;
  virtual void set_fit_intercept(bool fit_intercept);
  void sdca_primal_dual_relation(const double l_l2sq,
                                 const ArrayDouble &dual_vector,
                                 ArrayDouble &out_primal_vector);
};
typedef TModelGeneralizedLinear<double, std::atomic<double> > ModelGeneralizedLinearAtomicDouble;

%rename(ModelGeneralizedLinearAtomicFloat) TModelGeneralizedLinear<float, std::atomic<float> >;
class TModelGeneralizedLinear<float, std::atomic<float> > : public virtual TModelLabelsFeatures<float, std::atomic<float> >{
 public:
  ModelGeneralizedLinearAtomicFloat(const SBaseArrayFloat2dPtr features,
                         const SArrayFloatPtr labels,
                         const bool fit_intercept,
                         const int n_threads = 1);
  unsigned long get_n_coeffs() const override;
  virtual void set_fit_intercept(bool fit_intercept);
  void sdca_primal_dual_relation(const float l_l2sq,
                                 const ArrayFloat &dual_vector,
                                 ArrayFloat &out_primal_vector);
};
typedef TModelGeneralizedLinear<float, std::atomic<float> > ModelGeneralizedLinearAtomicFloat;
