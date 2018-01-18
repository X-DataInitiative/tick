// License: BSD 3 clause

%feature("autodoc", "1");  // 0 == no param types, 1 == show param types

%define %DocMeth(meth,def,details)
%feature("autodoc") meth def;
%feature("docstring") meth details;
%enddef

%define %DocClass(klass,doc)
%feature("docstring") klass doc;
%enddef

%include exception.i
%include stdint.i

typedef std::uint64_t ulong;

%{
  #include <system_error>
%}

%define EXCEPTION_ON
%exception {
    try {
        $action
    } catch (std::invalid_argument& e) {
      SWIG_exception_fail(SWIG_ValueError, e.what() );
    } catch (std::domain_error& e) {
      SWIG_exception_fail(SWIG_ValueError, e.what() );
    } catch (std::overflow_error& e) {
      SWIG_exception_fail(SWIG_OverflowError, e.what() );
    } catch (std::out_of_range& e) {
      SWIG_exception_fail(SWIG_IndexError, e.what() );
    } catch (std::length_error& e) {
      SWIG_exception_fail(SWIG_IndexError, e.what() );
    } catch (std::system_error& e) {
      SWIG_exception_fail(SWIG_RuntimeError, e.what() );
    } catch (std::runtime_error& e) {
      SWIG_exception_fail(SWIG_RuntimeError, e.what() );
    } catch (std::exception& e) {
      SWIG_exception_fail(SWIG_RuntimeError, e.what() );
    } catch (const std::string& str) {
      SWIG_exception_fail(SWIG_RuntimeError, str.c_str());
    }
}
%enddef

%define EXCEPTION_OFF
%exception;
%enddef

EXCEPTION_ON

