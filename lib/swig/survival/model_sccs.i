// License: BSD 3 clause

%include <std_shared_ptr.i>
%shared_ptr(ModelSCCS);

%{
#include "tick/survival/model_sccs.h"
%}

%include "model_lipschitz.i";

template <class T>
class TModelSCCS : public TModelLipschitz<T> {
 protected:
  //// this is unsupported currently in swig
  //using SBaseArrayT2dPtrList1D = std::vector<std::shared_ptr<BaseArray2d<T> > >;

 public:
  TModelSCCS(const std::vector<std::shared_ptr<BaseArray2d<T> > > &features,
             const SArrayIntPtrList1D &labels,
             const SBaseArrayULongPtr censoring, ulong n_lags);

  T loss(Array<T> &coeffs);

  void grad(Array<T> &coeffs, Array<T> &out);

  void compute_lip_consts();

  unsigned long get_rand_max();

  unsigned long get_epoch_size();

  // Number of parameters to be estimated. Can differ from the number of
  // features, e.g. when including an intercept.
  unsigned long get_n_coeffs() const;

  T get_lip_max();
};

%rename(ModelSCCS) TModelSCCS<double>;
class TModelSCCS<double> : public TModelLipschitz<double> {
 public:
  ModelSCCS(const SBaseArrayDouble2dPtrList1D &features,
                          const SArrayIntPtrList1D &labels,
                          const SBaseArrayULongPtr censoring,
                          ulong n_lags);
  double loss(ArrayDouble &coeffs);
  void grad(ArrayDouble &coeffs, ArrayDouble &out);
  void compute_lip_consts();
  unsigned long get_rand_max();
  unsigned long get_epoch_size();
  unsigned long get_n_coeffs() const;
  double get_lip_max();
};
typedef TModelSCCS<double> ModelSCCS;

%rename(ModelSCCSDouble) TModelSCCS<double>;
class TModelSCCS<double> : public TModelLipschitz<double> {
 public:
  ModelSCCSDouble(const SBaseArrayDouble2dPtrList1D &features,
                          const SArrayIntPtrList1D &labels,
                          const SBaseArrayULongPtr censoring,
                          ulong n_lags);
  double loss(ArrayDouble &coeffs);
  void grad(ArrayDouble &coeffs, ArrayDouble &out);
  void compute_lip_consts();
  unsigned long get_rand_max();
  unsigned long get_epoch_size();
  unsigned long get_n_coeffs() const;
  double get_lip_max();
};
typedef TModelSCCS<double> ModelSCCSDouble;

%rename(ModelSCCSFloat) TModelSCCS<float>;
class TModelSCCS<float> : public TModelLipschitz<float> {
 public:
  ModelSCCSFloat(const SBaseArrayFloat2dPtrList1D &features,
                          const SArrayIntPtrList1D &labels,
                          const SBaseArrayULongPtr censoring,
                          ulong n_lags);
  float loss(ArrayFloat &coeffs);
  void grad(ArrayFloat &coeffs, ArrayFloat &out);
  void compute_lip_consts();
  unsigned long get_rand_max();
  unsigned long get_epoch_size();
  unsigned long get_n_coeffs() const;
  float get_lip_max();
};
typedef TModelSCCS<float> ModelSCCSFloat;
