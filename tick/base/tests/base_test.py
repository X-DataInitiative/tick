# License: BSD 3 clause

# -*- coding: utf8 -*-
import unittest

from tick.base import Base
from tick.base.build.base import A0 as _A0

# Architecture of the classes
#         Base
#          |
#          A0
#         / \
#        A1 B1
#        |   |
#        A2  |
#        |   |
#        A3  |
#         \ /
#         A4


class A0(Base):
    """This is a class that inherit directly from Base

    Parameters
    ----------
    x0 : `int`
        This is doc of x0 from A0

    y0 : `float`
        This is doc of y0 from A0

    z0 : `float`
        This is doc of z0 from A0. z0 is not writable in A0 but will be
        writable in A1

    k0 : `float`
        This is doc of k0 from A0. k0 is writable in A0 but will not be
        writable in A1

    Attributes
    ----------
    creation_time : `float`
        This is doc of creation_time, an attribute which is readonly

    update_time : `float`
        This is doc of update_time, an attribute which is writable

    prop0 : `int`
        This is doc of a native property

    cpp_int : `int`
        This int exists in Python and in C++. When set in Python it must be
        changed in C++ as well
    """

    _attrinfos = {
        'x0': {
            'writable': False
        },
        'z0': {
            'writable': False
        },
        'k0': {
            'writable': True
        },
        'creation_time': {
            'writable': False
        },
        '_prop0': {},
        '_getter_called': {},
        '_setter_called': {},
        '_a0': {
            'writable': False
        },
        'cpp_int': {
            'cpp_setter': 'set_cpp_int'
        },
        'cpp_unlinked_int': {
            'cpp_setter': 'set_cpp_unlinked_int'
        },
    }

    _cpp_obj_name = "_a0"

    def __init__(self, x0: int, arg0, y0: float = 0., kwarg0='string'):
        Base.__init__(self)
        self.set_x0(x0)
        self.y0 = y0
        self.creation_time = self._get_now()
        self._prop0 = None
        # Two flags to know weather getter and setters have been called
        self._getter_called = False
        self._setter_called = False
        self._a0 = _A0()

        # Test that we can assign fields only declared in __init__
        self.arg0 = arg0
        self.kwarg0 = kwarg0

    def set_x0(self, x0):
        self.x0 = x0

    def force_set_x0(self, x0):
        self._set('x0', x0)

    @property
    def prop0(self):
        if self._prop0 is None:
            self._prop0 = 22
        self._getter_called = True
        return self._prop0

    @prop0.setter
    def prop0(self, val):
        self._setter_called = True
        self._prop0 = val

    @property
    def readonly_prop(self):
        return 44


class A1(A0):
    """This is a class that inherit from A0

    Parameters
    ----------
    x0 : `int`
        This is doc of x0 from A1 that overrides A0 doc

    y1 : `float`
        This is doc of y1 from A1

    z0 : `float`
        This is doc of z0 from A1. z0 is writable in A1 but was not
        writable in A0

    k0 : `float`
        This is doc of k0 from A1. k0 is not writable in A1 but was
        writable in A0
    """
    _attrinfos = {
        'z0': {
            'writable': True
        },
        'k0': {
            'writable': False
        },
    }

    def __init__(self, x0: int, y1: float = 1.):
        A0.__init__(self, x0, 22, y0=y1 / 2)
        self.set_x0(x0)
        self.y1 = y1


class B1(A0):
    """This is another class that inherit from A0
    """

    def __init__(self, x0: int, y0: float = 1.):
        A0.__init__(self, x0, 22, y0=y0)


class A2(A1):
    """This is an empty class that inherit from A1
    """

    def __init__(self, x0: int, y1: float = 1.):
        A1.__init__(self, x0, y1=y1)


class A3(A2):
    """This is an empty class that inherit from A2
    """

    def __init__(self, x0: int, y1: float = 1.):
        A2.__init__(self, x0, y1=y1)


class A4(A3, B1):
    """This is an empty class that inherit from A3 and B1
    """

    def __init__(self, x0: int, y1: float = 1.):
        A3.__init__(self, x0, y1=y1)
        B1.__init__(self, x0)


class Test(unittest.TestCase):
    def setUp(self):
        self.a0 = A0(2, 33)
        self.a1 = A1(10)
        self.a2 = A2(4)
        self.a4 = A4(3)

    def test_readonly(self):
        """...Test that assign read only attribute raises an error
        """

        def f(x):
            self.a0.x0 = x

        # Test we cannot set a readonly attribute in script
        self.assertRaises(AttributeError, f, 3)
        # Test we cannot set a readonly attribute from class method
        self.assertRaises(AttributeError, self.a0.set_x0, 32)
        # Test we can force setting a readonly attribute in script
        self.a0._set('x0', 32)
        # Test we can force setting a readonly attribute from class method
        self.a0.force_set_x0(32)

    def test_inherited_readonly(self):
        """...Test that assign read only attribute defined in parent class
        raises an error
        """

        def f(x):
            self.a1.x0 = x

        # Test we cannot set a readonly attribute in script
        self.assertRaises(AttributeError, f, 3)
        # Test we cannot set a readonly attribute from class method
        self.assertRaises(AttributeError, self.a1.set_x0, 32)
        # Test we can force setting a readonly attribute in script
        self.a1._set('x0', 32)
        # Test we can force setting a readonly attribute from class method
        self.a1.force_set_x0(32)

    def test_override_readonly(self):
        """...Test that assign read only attribute defined in parent class
        can be override in children class
        """

        def fz0(x):
            self.a0.z0 = x

        def fz1(x):
            self.a1.z0 = x

        def fz2(x):
            self.a2.z0 = x

        def fk0(x):
            self.a0.k0 = x

        def fk1(x):
            self.a1.k0 = x

        def fk2(x):
            self.a2.k0 = x

        def fx2(x):
            self.a2.x0 = x

        # This will raise any error as z0 is not writable in A0 and k0 is not
        #  writable in A1
        self.assertRaises(AttributeError, fz0, 3)
        self.assertRaises(AttributeError, fk1, 3)
        self.assertRaises(AttributeError, fk2, 3)
        self.assertRaises(AttributeError, fx2, 3)

        # This will not raise any error as z0 is writable in A1 and k0 is
        # writable in A0
        fz1(3)
        fk0(3)
        fz2(3)

    def test_unexisting_attribute(self):
        """...Test that getting or setting an unexisting attribute lead to an
        error
        """

        def getter():
            return self.a0.unexisting_attribute

        def setter(val):
            self.a0.unexisting_attribute = val

        self.assertRaises(AttributeError, getter)
        self.assertRaises(AttributeError, setter, 42)

        with self.assertRaises(AttributeError):
            self.a0._set('unexisting_attribute', 42)

    def test_deleteing_attribute(self):
        """...Test that deleting an attribute raises an error
        """

        def deleter():
            del self.a0.x0

        self.assertRaises(AttributeError, deleter)

    def test_attr_declared_in_doc(self):
        """...Test that an attribute declared in doc (but not in _attrinfos) is
        not unexisting
        """
        self.a0.update_time = self.a0._get_now()
        self.assertIsNotNone(self.a0.update_time)

    def test_cannot_access_unset(self):
        """...Test that an attribute that has not been set cannot be got and
        has correct error message
        """

        def f():
            return self.a0.update_time

        self.assertRaisesRegex(AttributeError,
                               "'A0' object has no attribute 'update_time'", f)

    def test_parameter_doc(self):
        """...Test that docstring is correctly parse for parameters
        """
        attrs_doc = {
            attr: infos.get('doc', [])
            for attr, infos in self.a0._attrinfos.items()
        }
        self.assertEqual(attrs_doc['x0'],
                         ['`int`', 'This is doc of x0 from A0', 'from A0'])
        self.assertEqual(attrs_doc['y0'],
                         ['`float`', 'This is doc of y0 from A0', 'from A0'])

    def test_attribute_doc(self):
        """...Test that docstring is correctly parse for attributes
        """
        attrs_doc = {
            attr: infos.get('doc', [])
            for attr, infos in self.a0._attrinfos.items()
        }
        self.assertEqual(attrs_doc['creation_time'], [
            '`float`', 'This is doc of creation_time, an attribute which '
            'is readonly', 'from A0'
        ])

    def test_inherited_parameter_doc(self):
        """...Test that docstring is correctly inherited

        ie. it is override if it is respecified and kept otherwise.
        """
        attrs_doc = {
            attr: infos.get('doc', [])
            for attr, infos in self.a1._attrinfos.items()
        }
        self.assertEqual(attrs_doc['x0'], [
            '`int`', 'This is doc of x0 from A1 that overrides A0 doc',
            'from A1'
        ])
        self.assertEqual(attrs_doc['y0'],
                         ['`float`', 'This is doc of y0 from A0', 'from A0'])
        self.assertEqual(attrs_doc['y1'],
                         ['`float`', 'This is doc of y1 from A1', 'from A1'])

    def test_multiple_inherited_parameter_doc(self):
        """...Test that docstring from the closest parent is kept

        In this case, A1 has index 3 (A4 -> A3 -> A2 -> A1) and A0 has index
        4 (through A1) even if it could have index 2 (through B1). We want to
        ensure that we keep doc from A1 instead of A0.
        """
        attrs_doc = {
            attr: infos.get('doc', [])
            for attr, infos in self.a4._attrinfos.items()
        }
        self.assertEqual(attrs_doc['x0'], [
            '`int`', 'This is doc of x0 from A1 that overrides A0 doc',
            'from A1'
        ])

    def test_multiple_inherited_attrinfo(self):
        """...Test that attrinfo from the closest parent is kept

        In this case, A1 has index 3 (A4 -> A3 -> A2 -> A1) and A0 has index
        4 (through A1) even if it could have index 2 (through B1). We want to
        ensure that we keep attrinfo from A1 instead of A0.
        """

        def fk4(x):
            self.a4.k0 = x

        def fz4(x):
            self.a4.z0 = x

        fz4(3)
        self.assertRaises(AttributeError, fk4, 3)

    def test_native_property_getter(self):
        """...Test that properties defined with the decorator @property can
        be got correctly
        """
        self.assertEqual(self.a0._getter_called, False)
        self.assertEqual(self.a0.prop0, 22)
        self.assertEqual(self.a0._getter_called, True)
        self.assertEqual(self.a0._setter_called, False)

    def test_native_property_setter(self):
        """...Test that properties defined with the decorator @property can
        be set correctly
        """
        self.assertEqual(self.a0._getter_called, False)
        self.assertEqual(self.a0._setter_called, False)
        self.a0.prop0 = 1
        self.assertEqual(self.a0._getter_called, False)
        self.assertEqual(self.a0._setter_called, True)
        self.assertEqual(self.a0.prop0, 1)
        self.assertEqual(self.a0._getter_called, True)

    def test_native_property_inherited_getter(self):
        """...Test that properties defined with the decorator @property can
        be got correctly when they are inherited
        """
        self.assertEqual(self.a1._getter_called, False)
        self.assertEqual(self.a1.prop0, 22)
        self.assertEqual(self.a1._getter_called, True)
        self.assertEqual(self.a1._setter_called, False)

    def test_cpp_setter(self):
        """...Test that a parameter that exists in both C++ and Python is
        correctly linked
        """
        # Test it works as expected with generic setter
        self.a0.cpp_int = 5
        self.assertEqual(self.a0.cpp_int, 5)
        self.assertEqual(self.a0._a0.get_cpp_int(), 5)

        # Test it works as expected with _set method
        self.a0._set('cpp_int', 15)
        self.assertEqual(self.a0.cpp_int, 15)
        self.assertEqual(self.a0._a0.get_cpp_int(), 15)

        # Test we raise correct error if setter was not a cpp method
        with self.assertRaisesRegex(
                NameError, "set_cpp_unlinked_int is not a method "
                "of.*"):
            self.a0.cpp_unlinked_int = 3

        with self.assertRaisesRegex(
                NameError, "set_cpp_unlinked_int is not a method "
                "of.*"):
            self.a0._set('cpp_unlinked_int', 3)

        # Test we raise correct error if _cpp_obj_name was not set
        del A0._cpp_obj_name
        with self.assertRaisesRegex(
                NameError, "_cpp_obj_name must be set as class "
                "attribute to use automatic C\+\+ setters"):
            self.a0.cpp_int = 15

        with self.assertRaisesRegex(
                NameError, "_cpp_obj_name must be set as class "
                "attribute to use automatic C\+\+ setters"):
            self.a0._set('cpp_int', 15)

    def test_cpp_inherited_setter(self):
        """...Test that a parameter that exists in both C++ and Python is
        correctly linked when they are inherited
        """
        self.a1.cpp_int = 5
        self.assertEqual(self.a1.cpp_int, 5)
        self.assertEqual(self.a1._a0.get_cpp_int(), 5)

    def test_getter_only_native_property(self):
        """...Test that native properties that have only a getter raise an
        error on setter
        """
        self.assertEqual(self.a0.readonly_prop, 44)

        def frop(x):
            self.a0.readonly_prop = x

        self.assertRaisesRegex(AttributeError, "can't set attribute", frop, 45)

    def test_attributes_are_not_shared(self):
        """...Test that two instances of the same class do not share any
        attributes"""
        a01 = A0(0, 1, y0=2, kwarg0='3')
        a02 = A0(10, 11, y0=12, kwarg0='13')
        self.assertEqual(a01.x0, 0)
        self.assertEqual(a01.arg0, 1)
        self.assertEqual(a01.y0, 2)
        self.assertEqual(a01.kwarg0, '3')
        self.assertEqual(a02.x0, 10)
        self.assertEqual(a02.arg0, 11)
        self.assertEqual(a02.y0, 12)
        self.assertEqual(a02.kwarg0, '13')


if __name__ == "__main__":
    unittest.main()
