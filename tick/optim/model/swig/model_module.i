%module model

%include defs.i
%include std_shared_ptr.i

%shared_ptr(Model);
%shared_ptr(ModelLabelsFeatures);
%shared_ptr(ModelGeneralizedLinear);
%shared_ptr(ModelGeneralizedLinearWithIntercepts);
%shared_ptr(ModelLipschitz);
%shared_ptr(ModelLinReg);
%shared_ptr(ModelLinRegWithIntercepts);
%shared_ptr(ModelLogReg);
%shared_ptr(ModelPoisReg);

%shared_ptr(ModelHawkes);

%shared_ptr(ModelHawkesSingle);
%shared_ptr(ModelHawkesFixedExpKernLogLik);
%shared_ptr(ModelHawkesFixedExpKernLeastSq);
%shared_ptr(ModelHawkesFixedSumExpKernLeastSq);

%shared_ptr(ModelHawkesList);
%shared_ptr(ModelHawkesLeastSqList);
%shared_ptr(ModelHawkesFixedExpKernLeastSqList);
%shared_ptr(ModelHawkesFixedSumExpKernLeastSqList);
%shared_ptr(ModelHawkesFixedExpKernLogLikList);

%shared_ptr(ModelCoxRegPartialLik);
%shared_ptr(ModelSCCS);

%{
#include "tick_python.h"
%}

%import(module="tick.base") base_module.i

%include model.i

%include model_labels_features.i

%include model_lipschitz.i

%include model_generalized_linear.i

%include model_generalized_linear_with_intercepts.i

%include hawkes.i

%include linreg.i

%include linreg_with_intercepts.i

%include logreg.i

%include poisreg.i

%include coxreg_partial_lik.i

%include sccs.i