// License: BSD 3 clause

%module hawkes_model

%include tick/base/defs.i
%include tick/base/serialization.i
%include <std_shared_ptr.i>

%{
#include "tick/base/tick_python.h"
%}

%import(module="tick.base") tick/base/base_module.i

%include tick/base_model/base_model_module.i

%shared_ptr(ModelHawkes);
%shared_ptr(ModelHawkesList);
%shared_ptr(ModelHawkesLeastSq);
%shared_ptr(ModelHawkesLogLik);

%shared_ptr(ModelHawkesExpKernLeastSq);
%shared_ptr(ModelHawkesSumExpKernLeastSq);
%shared_ptr(ModelHawkesExpKernLogLik);
%shared_ptr(ModelHawkesSumExpKernLogLik);


%include base/model_hawkes.i
%include base/model_hawkes_list.i
%include base/model_hawkes_leastsq.i
%include base/model_hawkes_loglik.i

%include list_of_realizations/model_hawkes_expkern_leastsq.i
%include list_of_realizations/model_hawkes_sumexpkern_leastsq.i
%include list_of_realizations/model_hawkes_expkern_loglik.i
%include list_of_realizations/model_hawkes_sumexpkern_loglik.i

%shared_ptr(ModelCustomBasic);
%shared_ptr(ModelHawkesCustom);
%shared_ptr(ModelHawkesCustomType2);
%shared_ptr(ModelHawkesSumExpCustom);
%shared_ptr(ModelHawkesSumExpCustomType2);


%include base/modelcustombasic.i

%include list_of_realizations/hawkes_fixed_expkern_loglik_custom.i
%include list_of_realizations/hawkes_fixed_expkern_loglik_custom2.i

%include list_of_realizations/hawkes_fixed_sumexpkern_loglik_custom.i
%include list_of_realizations/hawkes_fixed_sumexpkern_loglik_custom2.i