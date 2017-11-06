
%module proj

%include defs.i

%{
#include "tick_python.h"
%}

%import(module="tick.base") base_module.i

%include "proj.i"

