
%include defs.i
%module proj


%{
#include "tick_python.h"
%}

%import(module="tick.base") base_module.i


%include proj.i

