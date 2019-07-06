
.. _dev:

Developer documentation
=======================

.. todo::

    This page might need some update


Main highlights
---------------

* **Python 3 only**!
* Fast, most computation are done in C++11 (including multi-threading)
* Interplay between Python and C++ is done using swig


.. contents::
    :depth: 3
    :backlinks: none

.. _BaseClass:

Base class
----------
Most of our objects inherit from `tick.base.Base` class. If this class does
not have many public methods, it allow us to benefit from several behaviors.
These behaviors are deducted from a private dictionary set by developer :
`_attrinfos`.

.. testsetup:: *

    from tick.base import Base

Read-only attributes
^^^^^^^^^^^^^^^^^^^^

Behavior
~~~~~~~~
The first feature is the ability to set read-only attributes. If you think
end-user should not modify an attribute of your class, you can simply set
writable to False (writable is True by default).

.. testcode:: [readonly]

    class A(Base):
        _attrinfos = {
            'readonly_attr' : {
                'writable' : False
            }
        }

    a = A()

Then if you try to set it

.. doctest:: [readonly]

    >>> a.readonly_attr = 3
    Traceback (most recent call last):
      ...
    AttributeError: readonly_attr is readonly in A

How to set it manually?
~~~~~~~~~~~~~~~~~~~~~~~
First if you set a read-only attribute during the initialization phase (ie.
in `__init__` method or in a method called by it) you will not raise any error.

.. testcode:: [readonly]

    class A(Base):
        _attrinfos = {
            'readonly_attr' : {'writable' : False}
        }

        def __init__(self, readonly_value):
            self.readonly_attr = readonly_value

This code snippet will work as usual.

.. doctest:: [readonly]

    >>> a = A(5)
    >>> a.readonly_attr
    5

But if you need to change this attribute after the initialization phase, you
can force set it by using `_set` method.


.. doctest:: [readonly]

    >>> a._set('readonly_attr', 10)
    >>> a.readonly_attr
    10

Settable attributes
^^^^^^^^^^^^^^^^^^^
Base class also restricts which attributes can be set. Attributes that can be
set are:

* Attributes contained in `_attrinfos` dictionary
* Attributes documented in class docstring (with numpydoc style)
* Attributes passed as argument to `__init__`

.. warning::

    When you document an attribute with numpydoc style, do not forget the
    space before the colon that follow its name.

Hence, if an attribute was never mentioned in your class before, trying to
set it will raise an exception.

.. testcode:: [settable]

    class A(Base):
        """This is an awesome class that inherits from Base

        Parameters
        ----------
        documented_parameter : `int`
            This is a documented parameter of my class

        Attributes
        ----------
        documented_attribute : `string`
            This is a documented attribute of my class
        """
        _attrinfos = {
            'attr_in_attrinfos' : {}
        }
        def __init__(self, documented_parameter, undocumented_parameter):
            pass

The following will work as expected

.. doctest:: [settable]

    >>> a = A(10, 12)
    >>> a.documented_parameter = 32
    >>> a.documented_attribute = 'bananas'
    >>> a.undocumented_parameter = 'are too many'
    >>> a.documented_parameter, a.documented_attribute, a.undocumented_parameter
    (32, 'bananas', 'are too many')

But this raises an error

.. doctest:: [settable]

    >>> a = A(10, 12)
    >>> a.unexisting_attr = 25
    Traceback (most recent call last):
     ...
    AttributeError: 'A' object has no settable attribute 'unexisting_attr'


Link with C++ setter
^^^^^^^^^^^^^^^^^^^^
Another useful feature is the possibility to add a direct linking between a
Python attribute and its C++ equivalent.

In many cases our code consists in a Python object which encompasses a C++
object used for intense computations. Find more details in the SWIG part
of this documentation. In this setting we might want to update our C++ object
each time our Python object is. We can do so by specifying which setter to
call when an attribute is modified in Python.

For this example, let's suppose we have a C++ class (named `_A`) that has a int
attribute associated to a setter (`set_cpp_int`) and a getter (`get_cpp_int`).
In order to enable the linking we must specify:

* What is the C++ object's name, through `_cpp_obj_name` attribute of the class
* What is the C++ method that sets attribute `cpp_int`, through `cpp_setter`
  in `_attrinfos` dictionary

.. testsetup:: [cpp_setter]

    from tick.base.build.base import A0 as _A

.. testcode:: [cpp_setter]

    class A(Base):
        _attrinfos = {
            'cpp_int': {'cpp_setter': 'set_cpp_int'},
            '_a' : {'writable' : False}
        }
        _cpp_obj_name = "_a"

        def __init__(self):
            self._a = _A()
            self.cpp_int = 0

Now each time we will modify `cpp_int` attribute of an instance of the class
`A`, `set_cpp_int` method of the C++ object will be called and modify the
value of the C++ int.

.. doctest:: [cpp_setter]

    >>> a = A()
    >>> a.cpp_int, a._a.get_cpp_int()
    (0, 0)
    >>> a.cpp_int = -4
    >>> a.cpp_int, a._a.get_cpp_int()
    (-4, -4)

.. note::
    If the reader wants to run this example, he might find the corresponding
    class by importing it `from tick.base.utils.build.utils import A0 as _A`.

How does this work?
^^^^^^^^^^^^^^^^^^^
This class behavior is obtained thanks to Python metaclasses. A metaclass is
the object that is called to create the class object itself. For example, it
allow us to automatize property creation. For more information, please report
to `Python documentation`_.

What we do is creating a property for each attribute. This property is linked
to a hidden attribute, stored with the same name of the property with a
double underscore before

.. _Python documentation:
    https://docs.python.org/3/reference/datamodel.html#
    customizing-class-creation

If we create the following class `A`:

.. testcode:: [how]

    class A(Base):
        def __init__(self, attr):
            self.attr = attr

We have access to the property `attr` and its linked attribute `__attr`:

.. doctest:: [how]

    >>> a = A(15)
    >>> a.attr, a.__attr
    (15, 15)

Two good practises to avoid unexpected behaviors:

* Do not define an attribute that starts with a double underscore
* Add property documentation in class docstring instead of property getter


How to add a new model, solver or prox
--------------------------------------

Many of our models, prox and solvers are Python classes that wraps a C++ class
which handles the heavy computations. This allows us to have a code that runs
fast.

Let's see what we should do if we want to add prox L2. Adding a model or a
solver is basically identical.

Create C++ class
^^^^^^^^^^^^^^^^

First we need to create the C++ class that will be wrapped by our Python
class later. We want our prox to be able to give the value of the
penalization at a given point and call the proximal operator on a given vector.

Here is what our .h file should look like

.. code-block:: cpp

    class ProxL2Sq {

    protected:
        double strength;

    public:
        ProxL2Sq(double strength);
        double value(ArrayDouble &coeffs) const;
        void call(ArrayDouble &coeffs, double step, ArrayDouble &out) const;
        inline void set_strength(double strength){
            this->strength = strength;
        }
    };

Basically we have one constructor that set the only one parameter strength
(usually denoted by lambda), and the two methods we described above.

Our .cpp implementation looks like:

.. code-block:: cpp

    #include "prox_l2sq.h"

    ProxL2Sq::ProxL2Sq(double strength) {
        this->strength = strength;
    }

    double ProxL2Sq::value(ArrayDouble &coeffs) const {
        return 0.5 * coeffs.normsq();
    }

    void ProxL2Sq::call(ArrayDouble &coeffs, double step, ArrayDouble &out) const {
        for (unsigned long i; i < coeffs.size; ++i)
            out[i] = coeffs[i] / (1 + step * strength);
    }

In tick these files are stored in the lib/cpp and lib/include folders

Link it with Python
^^^^^^^^^^^^^^^^^^^

Create SWIG file
~~~~~~~~~~~~~~~~

Now that our proximal operator is defined in C++ we need to make it available
in Python. We do it thanks to `SWIG <http://www.swig.org/Doc4.0/>`_.
Hence we have to create a .i file. In tick we store them in the lib/swig folder.

This .i file looks a lot like our .h file.

.. code-block:: cpp

    %include <std_shared_ptr.i>
    %shared_ptr(ProxL2Sq);

    %{
    #include "prox_l2sq.h"
    %}

    class ProxL2Sq {
    public:
        ProxL2Sq(double strength);
        double value(ArrayDouble &coeffs) const;
        void call(ArrayDouble &coeffs, double step, ArrayDouble &out) const;
        virtual void set_strength(double strength);
    };

In this file our goal is to explain to Python what it can do with this class. In
our example it will be able to instantiate it by calling its constructor
with a double, and call three methods, `value`, `call` and `set_strength`.

.. note::
  * There is no interest in mentioning here any private method or attribute of
    the class as this is what Python see and Python would not be able to call
    them.
  * We need to include the file in which is declared the class we are talking
    about in the .i file. This what we do with `#include "prox_l2sq.h"`.
  * Finally, as we will want to share our proximal operator and as it might be
    used by several objects, we wrap it in class from the standard library: the
    shared pointer. To make SWIG aware that this class will be used with shared
    pointers we must add `%shared_ptr(ProxL2Sq);` which must be done after
    `%include \<std_shared_ptr.i\>`.

.. note::
  In tick our ProxL2Sq class is not really identical as it inherits
  from Prox abstract class. Hence some of this logic might not be present in
  the exact same file. Everything that concerns prox, is imported through
  `prox_module.i`.

Reference it in Python build
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Now that we have written our .i file we should add our files to our python
script that builds the extension : `setup.py`.

Most of the time, we add a file that belongs to a module that has already
been created. In this case we only need to add its source files at the right
place in `setup.py`.

Let's supposed we already had two prox in our module (abstract class Prox and
prox L1), we need to add `prox_l2sq.cpp` and `prox_l2sq.h` that we have just
created at the following place.

.. code-block:: python
    :emphasize-lines: 4, 7

    prox_core_info = {
        "cpp_files": ["prox.cpp",
                      "prox_l1.cpp",
                      "prox_l2sq.cpp"],
        "h_files": ["prox.h",
                    "prox_l1.h",
                    "prox_l2sq.h"],
        "swig_files": ["prox_module.i", ],
        "module_dir": "./tick/optim/prox/",
        "extension_name": "prox",
        "include_modules": base_modules
    }

.. note::
  We do not need to add `prox_l2sq.i` file here as it is imported
  in `prox_module.i` with a `%include` operator. This operator works like a
  copy/paste of the code of the included file.

Use class in Python
^^^^^^^^^^^^^^^^^^^

Now that our C++ class is linked with Python we can import it and use its
methods that we have declared in the .i file.

In tick we always wrap C++ classes in a Python class that will call C++
object methods when it needs to perform the computations. Hence here is the
Python class we might create:

.. code-block:: python

    import numpy as np
    from .build.prox import ProxL2Sq as _ProxL2sq

    class ProxL2Sq:
        _attrinfos = {
            "strength": {
                "cpp_setter": "set_strength"
            }
        }
        _cpp_obj_name = "_prox"

        def __init__(self, strength: float):
            self._prox = _ProxL2sq(strength)
            self.strength = strength

        def value(self, coeffs: np.ndarray):
            return self._prox.value(coeffs)

You might have seen that we instantiate a dictionary called `_attrinfos` in
the class declaration. This dictionary is useful in many ways and you should
refer to BaseClass_. Here we use one of
its functionalities: automatic set of C++ attributes. Each time `strength` of
our prox will be modified, the `set_strength` method of the object stored in
`_prox` attribute (as specified by `_cpp_obj_name` will be called with the
new value passed as argument). This allow us to have strength values of
Python and C++ that are always linked.

Enable Python-pickling of C++ objects
-------------------------------------

In some cases we need our C++-wrapped objects to be picklable by Python. For
instance, if we need to use the object as part of the Python `multiprocessing`
library, or if we want to implement some stop/resume functionality.

The way it has been done in a number of existing `tick` classes is by
(de)serializing the object to and from string-types via the
`Cereal <http://uscilab.github.io/cereal/>`_ library.

In example:

.. code-block:: cpp

    #include <cereal/types/polymorphic.hpp>
    #include <cereal/types/base_class.hpp>

    class HawkesKernelSumExp : public HawkesKernel {
     public:
      ...

      template <class Archive>
      void serialize(Archive & ar) {
        ar(cereal::make_nvp("HawkesKernel", cereal::base_class<HawkesKernel>(this)));

        ar(CEREAL_NVP(use_fast_exp));
        ar(CEREAL_NVP(n_decays));
        ar(CEREAL_NVP(intensities));
        ar(CEREAL_NVP(decays));
        ar(CEREAL_NVP(last_convolution_time));
        ar(CEREAL_NVP(last_convolution_values));
        ar(CEREAL_NVP(convolution_restart_index));
      }

      ...
    };

We add the serialize method, and in the method body we specify which members of
the class to put into the serialization archive. Note that members are wrapped
with `CEREAL_NVP()`, which is a Cereal macro to add the value and *name* of a
variable. The standard tick classes such as `ArrayDouble` can be added as
archive members without additional effort (e.g. intensities and decays in the
above example).

In the example above we also add the values of the base class (in this case
`HawkesKernel`). Here we manually specify the value name with
`cereal::make_nvp`.

This takes care of the C++-part of serialization. We add Pickle functionality
directly in the SWIG interface file:

.. code-block:: cpp

    %{
    #include "hawkes.h"
    %}

    %include serialization.i

    class HawkesKernelSumExp : public HawkesKernel {
     public:
      ...

      HawkesKernelSumExp();

      ...
    };

    TICK_MAKE_PICKLABLE(HawkesKernelSumExp);

A convenience macro `TICK_MAKE_PICKLABLE` is available to add all the necessary
bits to a SWIG definition in order to make it picklable in Python.

`TICK_MAKE_PICKLABLE` takes any number of arguments. The first being the class
name of the class to be pickled. Any following arguments will be forwarded to
the Python constructor of the class for initialization (used when
unpickling/reconstructing an object). In the example above, no parameters are
given to the constructor.

The macro adds a block of Python code with a `__getstate__` method to return a
serialized copy of the object, and a `__setstate__` method to reconstruct the
object from a string value (this is where the initialization/constructor
parameters play in).

Similarly, `TICK_MAKE_TEMPLATED_PICKLABLE` is provided if needed to specify
a type with templated parameters.

It's important to consider the initialization of the Python object. In some
cases it might be convenient to add a parameter-less C++ constructor that
initializes an empty object. Otherwise existing constructors should be used.

Now that the Python class has methods to get/set the object state, the pickle
module may work on the class.

Splitting serialization and deserialization methods
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
In some cases it's convenient to split the serialization method into two; one
for loading (deserializing) and saving (serializing) an object. For example, if
an archive member needs complex initialization during loading, it may be easier
to have this done in a separate method.

In Cereal, we can define load and save methods separately:

.. code-block:: cpp

    template <class Archive>
    void save(Archive & ar) const {
      ar(x);
      ar(y);
      ar(z.get_foo());
    }

    template <class Archive>
    void load(Archive & ar) {
      ar(x);
      ar(y);

      float temp = 0.0f;
      ar(temp);

      z = Z(temp);
    }

See `Cereal website <https://uscilab.github.io/cereal/serialization_functions
.html>`_
for more details.

Serializing class hierarchies
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
When serializing a class that is part of a hierarchy, it's usually sufficient to
add the base class as one of the archive members, as shown in the example:

.. code-block:: cpp

    template <class Archive>
    void serialize(Archive & ar) {
      ...
      ar(cereal::make_nvp("HawkesKernel", cereal::base_class<HawkesKernel>(this)));
      ...
    }

However, if some base class in the hierarchy defines split save/load methods,
and a derived class defines a single serialize method (or vice versa), it may be
necessary to inform Cereal which serialization method(s) to use. Cereal provides
a macro to achieve this:

.. code-block:: cpp

    // Always use the single 'serialize' method when serializing the Hawkes class
    CEREAL_SPECIALIZE_FOR_ALL_ARCHIVES(Hawkes, cereal::specialization::member_serialize)

    // OR

    // Always use the split 'load/save' methods when serializing the Hawkes class
    CEREAL_SPECIALIZE_FOR_ALL_ARCHIVES(Hawkes, cereal::specialization::member_load_save)

The macros need to be put after the definition of the classes, and in the global
namespace (i.e. not in tick).

Serializing smart pointers
^^^^^^^^^^^^^^^^^^^^^^^^^^
Serializing smart pointers such as `std::shared_ptr` or `std::unique_ptr` is supported
by Cereal with a minimum of needed work.

An archive member of a smart pointer type is tagged with additional information
about the actual value type when serialized. For this to happen, Cereal needs to be
informed of the available derived classes of the base class that is serialized.

We do this in the following way:

.. code-block:: cpp

    CEREAL_REGISTER_TYPE(DerivedClass)

With this macro in place, Cereal will be able to save and restore values with the
correct polymorphic types. You can see more in the
`Cereal documentation on polymorphic types <https://uscilab.github.io/cereal/polymorphism.html>`_.

Tips for debugging C++ classes in tick
--------------------------------------

In the header file `debug.h` we have a number of macro definitions to aid in the development of the tick library.

Debug output
^^^^^^^^^^^^
For the sake of convenience, `debug.h` defines `TICK_DEBUG()` and
`TICK_WARNING()` to print messages to stdout or stderr respectively. They are
both used as streaming interfaces:

.. code-block:: cpp

    ArrayDouble arr = f();

    TICK_DEBUG() << "Printing an array to stdout: " << arr;
    TICK_WARNING() << "Printing an array to stderr: " << arr;

Most types that can be inserted into std::stringstream can also be inserted into
these interfaces. Notice that `tick` arrays can also be inserted.

Raising errors
^^^^^^^^^^^^^^
To generate errors there is an similar macro which will raise a C++ exception to
be caught by Python. The interface is almost identical to `TICK_DEBUG()` and
`TICK_WARNING()` except that the input must be placed within the parenthesis:

.. code-block:: cpp

    ArrayDouble arr = f();

    TICK_ERROR("A fatal error occurred because of this array: " << arr);

This will throw an exception which (if used via the Python interface) will be
caught in the SWIG interface layer and raised as an error in Python.

The exception thrown can include a backtrace to the point of error. For this to
happen, the compilation of the library must include the DEBUG_VERBOSE flag
(see `setup.py`).

Deprecation
^^^^^^^^^^^
The library is under continuous development and occasionally some internal
implementations will be phased out. To ease this process the macro
`TICK_DEPRECATED` is useful to mark variables or definition as no-longer-fit to
use. Code will still compile and link, but warnings will be generated:
::

    .../tick/deprecated.cpp: In member function ‘void f()’:
    .../tick/deprecated.cpp:20:3: warning: ‘int some_method()’ is deprecated (declared at .../deprecated.cpp:10) [-Wdeprecated-declarations]
