// License: BSD 3 clause

%module preprocessing

%include std_shared_ptr.i
%include tick/base/defs.i
#include tick/base/serialization.h

%shared_ptr(SparseLongitudinalFeaturesProduct);
%shared_ptr(LongitudinalFeaturesLagger);

%{
#include "tick/base/tick_python.h"
%}

%import(module="tick.base") tick/base/base_module.i

%include sparse_longitudinal_features_product.i
%include longitudinal_features_lagger.i