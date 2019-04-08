// License: BSD 3 clause

%module crandom
%include tick/base/defs.i

%{
#include "tick/base/tick_python.h"
%}

%import(module="tick.base") tick/base/base_module.i

%{
#include "tick/random/test_rand.h"
%}

%include "tick/random/test_rand.h"
