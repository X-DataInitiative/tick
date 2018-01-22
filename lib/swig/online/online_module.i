// License: BSD 3 clause

%module online

%include defs.i
%include std_shared_ptr.i
%include serialization.i

%shared_ptr(OnlineForestRegressor);
%shared_ptr(OnlineForestClassifier);

%{
#include "tick/base/tick_python.h"
%}

%import(module="tick.base") base_module.i

%include online_forest_regressor.i
%include online_forest_classifier.i
