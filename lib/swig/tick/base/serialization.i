// License: BSD 3 clause

%include std_string.i

%{
#include "tick/base/serialization.h"
%}
// TICK_SWIG_INCLUDE is a flag as the MACROs to disable certain compiler errors
// messes with swig
%define TICK_SWIG_INCLUDE
%include tick/base/serialization.h
%enddef TICK_SWIG_INCLUDE

%define TICK_MAKE_PICKLABLE(CLASS_NAME, CONSTRUCTOR_ARGS...)

  %template(##CLASS_NAME##Deserialize) tick::object_from_string<CLASS_NAME>;
  %template(##CLASS_NAME##Serialize) tick::object_to_string<CLASS_NAME>;

  %extend CLASS_NAME {
    %pythoncode {
            def __getstate__(self): return CLASS_NAME##Serialize(self)
            def __setstate__(self, s):
                self.__init__(CONSTRUCTOR_ARGS)
                return CLASS_NAME##Deserialize(self, s)
    }
  }
%enddef

%define TICK_MAKE_TEMPLATED_PICKLABLE(T_CLASS, I_CLASS, T, CONSTRUCTOR_ARGS...)

  %template(##I_CLASS##Deserialize) tick::object_from_string<I_CLASS>;
  %template(##I_CLASS##Serialize) tick::object_to_string<I_CLASS>;

  %extend T_CLASS<T> {
    %pythoncode {
            def __getstate__(self): return I_CLASS##Serialize(self)
            def __setstate__(self, s):
                self.__init__(CONSTRUCTOR_ARGS)
                return I_CLASS##Deserialize(self, s)
    }
  }
%enddef

%define TICK_MAKE_TK_PICKLABLE(T_CLASS, I_CLASS, T, K, CONSTRUCTOR_ARGS...)

  %template(##I_CLASS##Deserialize) tick::object_from_string<I_CLASS>;
  %template(##I_CLASS##Serialize) tick::object_to_string<I_CLASS>;

  %extend T_CLASS<T, K> {
    %pythoncode {
            def __getstate__(self): return I_CLASS##Serialize(self)
            def __setstate__(self, s):
                self.__init__(CONSTRUCTOR_ARGS)
                return I_CLASS##Deserialize(self, s)
    }
  }
%enddef

