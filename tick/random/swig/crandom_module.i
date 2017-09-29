// License: BSD 3 clause

%module crandom
%include defs.i

%{
#include "tick_python.h"
%}

%import(module="tick.base") base_module.i

%{
#include "test_rand.h"
%}

%include "test_rand.h"
