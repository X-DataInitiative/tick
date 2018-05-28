# License: BSD 3 clause

# import warnings
import os
import inspect
from datetime import datetime
from abc import ABCMeta
import json
import pydoc
import numpy as np
import numpydoc as nd
from numpydoc import docscrape
import copy

# The metaclass inherits from ABCMeta and not type, since we'd like to
# do abstract classes in tick that inherits from ABC

# TODO: readonly attributes cannot be linked to c++ setters


class BaseMeta(ABCMeta):
    # Default behaviour of an attribute is writable, with no C++ setter
    default_attrinfo = {"writable": True, "cpp_setter": None}
    default_classinfo = {
        'is_prop': False,
        'in_doc': False,
        'doc': [],
        'in_init': False
    }

    @staticmethod
    def hidden_attr(attr_name):
        return '__' + attr_name

    @staticmethod
    def set_cpp_attribute(self, val, cpp_setter):
        """ Set the linked cpp attribute if possible

        Parameters
        ----------
        self : `object`
            the instance

        val : `object`
            the value to be set

        cpp_setter : `function`
            the function to use
        """
        # First we get the C++ object from its name (its name is an attribute
        # in the class)
        if not hasattr(self, "_cpp_obj_name"):
            raise NameError("_cpp_obj_name must be set as class attribute to "
                            "use automatic C++ setters")

        # Retrieve C++ associated object if it has been instantiated
        cpp_obj = None
        if hasattr(self, self._cpp_obj_name):
            cpp_obj = getattr(self, self._cpp_obj_name)

        # If the cpp_obj is instantiated, we update it
        if cpp_obj is not None:
            # Get the setter for this attribute in the C++
            if not hasattr(cpp_obj, cpp_setter):
                raise NameError("%s is not a method of %s" %
                                (cpp_setter, cpp_obj.__class__))
            cpp_obj_setter = getattr(cpp_obj, cpp_setter)
            cpp_obj_setter(val)

    @staticmethod
    def detect_if_called_in_init(self):
        """This function examine stacktrace in order to determine if it has
        been called from the __init__ function of the given instance

        Parameters
        ----------
        self : `object`
            The instance

        Returns
        -------
        set_int_init : `bool`
            True if this function was called by __init__
        """
        # It is forbidden to set a readonly (non writable) attribute
        # expect from __init__ function of the class
        set_in_init = False
        # trace contains information of the current execution
        # environment
        trace = inspect.currentframe()
        while trace is not None:
            # We retrieve the name of the executor (for example the
            # function that launched this command)
            exec_name = trace.f_code.co_name
            # We inspect the local variables
            if 'self' in trace.f_locals:
                local_self = trace.f_locals['self']
            else:
                local_self = None
            # We check if environment corresponds to our instance's
            # __init__
            if exec_name == '__init__' and local_self == self:
                set_in_init = True
                break
            # If this frame was not the good one, we try the previous
            # one, the one that has launched it
            # If there is no previous one, `None` will be returned
            trace = trace.f_back

        return set_in_init

    @staticmethod
    def build_property(class_name, attrs, attr_name, writable, cpp_setter):
        """
        Builds a property

        Parameters
        ----------
        class_name : `str`
            Name of the class

        attrs : `dict`
            The attributes of the class

        attr_name : `str`
            Name of the attribute for which we build a property

        writable : `bool`
            If True, we attribute can be changed by the user. If not,
            then an error will be raise when trying to change it.

        override : `bool`
            Not implemented yet

        cpp_setter : `str` or `None`
            Name of the setter in the c++ object embedded in the class
            for the attribute

        Returns
        -------
        output : property
            The required property
        """

        hidden_name = BaseMeta.hidden_attr(attr_name)

        def getter(self):
            # if it was not assigned yet we raise the correct error message
            if not hasattr(self, hidden_name):
                raise AttributeError("'%s' object has no attribute '%s'" %
                                     (class_name, attr_name))
            # Get the attribute
            return object.__getattribute__(self, hidden_name)

        def create_base_setter():
            if cpp_setter is None:
                # There is no C++ setter, we just set the attribute
                def setter(self, val):
                    object.__setattr__(self, hidden_name, val)
            else:
                # There is a C++ setter to apply
                def setter(self, val):
                    object.__setattr__(self, hidden_name, val)
                    # We update the C++ object embedded in the class
                    # as well.
                    BaseMeta.set_cpp_attribute(self, val, cpp_setter)

            return setter

        base_setter = create_base_setter()
        if writable:
            setter = base_setter
        else:
            # If it is not writable we wrap the base setter with something
            # that detect if attribute setting was called in __init__
            def setter(self, val):
                set_in_init = Base.detect_if_called_in_init(self)
                # If and only if this was launched from our instance's __init__
                # we allow user to set the attribute
                if set_in_init:
                    base_setter(self, val)
                else:
                    raise AttributeError(
                        "%s is readonly in %s" % (str(attr_name), class_name))

        def deletter(self):
            raise AttributeError(
                "can't delete %s in %s" % (str(attr_name), class_name))

        # We set doc to None otherwise it will interfere with
        # the docstring of the class.
        # This is very useful as we can have bugs with sphinx
        # otherwise (conflict between an attribute's docstring
        # and a docstring of a property with the same name).
        # All attributes are actually properties when the base
        # class is Base.
        # The docstring of all properties are then putted back
        # in the __init__ of the Base class below.
        prop = property(getter, setter, deletter, None)
        return prop

    @staticmethod
    def create_property_doc(class_name, attr_doc):
        """Create doc that will be attached to property

        Parameters
        ----------
        class_name : `str`
            Name of the class the property comes from

        attr_doc : `list`
            List output by numpydoc contained parsed documentation

        Returns
        -------
        The formatted doc
        """
        attr_type = attr_doc[1]
        attr_docstring = [
            line for line in attr_doc[2] if len(line.strip()) > 0
        ]
        attr_from = 'from %s' % class_name

        doc = [attr_type] + attr_docstring + [attr_from]
        return doc

    @staticmethod
    def find_init_params(attrs):
        """Find the parameters passed to the class's __init__
        """
        ignore = ['self', 'args', 'kwargs']

        # if class has no __init__ method
        if "__init__" not in attrs:
            return []

        return [
            key
            for key in inspect.signature(attrs["__init__"]).parameters.keys()
            if key not in ignore
        ]

    @staticmethod
    def find_properties(attrs):
        """Find all native properties of the class
        """
        return [
            attr_name for attr_name, value in attrs.items()
            if isinstance(value, property)
        ]

    @staticmethod
    def find_documented_attributes(class_name, attrs):
        """Parse the documentation to retrieve all attributes that have been
        documented and their documentation
        """
        # If a class is not documented we return an empty list
        if '__doc__' not in attrs:
            return []

        current_class_doc = inspect.cleandoc(attrs['__doc__'])
        parsed_doc = docscrape.ClassDoc(None, doc=current_class_doc)
        attr_docs = parsed_doc['Parameters'] + parsed_doc['Attributes'] + \
            parsed_doc['Other Parameters']

        attr_and_doc = []

        create_property_doc = BaseMeta.create_property_doc
        for attr_doc in attr_docs:
            attr_name = attr_doc[0]
            if ':' in attr_name:
                raise ValueError("Attribute '%s' has not a proper "
                                 "documentation, a space might be missing "
                                 "before colon" % attr_name)
            attr_and_doc += [(attr_name,
                              create_property_doc(class_name, attr_doc))]
        return attr_and_doc

    @staticmethod
    def extract_attrinfos(class_name, attrs):
        """Inspect class attrs to create aggregate all attributes info of the
        current class

        In practice, we inspect documented attributes, properties,
        parameters given to __init__ function and finally what user has
        filled in _attrinfos

        Parameters
        ----------
        class_name : `str`
            The name of the class (needed to create associated doc)

        atts : `dict`
            Dictionary of all futures attributes of the class

        Returns
        -------
        current_attrinfos : `dict`
            Subdict of the global classinfos dict concerning the attributes
            of the current class.
        """
        current_attrinfos = {}

        # First we look at all documented attributes
        for attr_name, attr_doc in \
                BaseMeta.find_documented_attributes(class_name, attrs):
            current_attrinfos.setdefault(attr_name, {})
            current_attrinfos[attr_name]['in_doc'] = True
            current_attrinfos[attr_name]['doc'] = attr_doc

        # Second we look all native properties
        for attr_name in BaseMeta.find_properties(attrs):
            current_attrinfos.setdefault(attr_name, {})
            current_attrinfos[attr_name]['is_prop'] = True

        # Third we look at parameters given to __init__
        for attr_name in BaseMeta.find_init_params(attrs):
            current_attrinfos.setdefault(attr_name, {})
            current_attrinfos[attr_name]['in_init'] = True

        # Finally we use _attrinfos provided dictionary
        attrinfos = attrs.get("_attrinfos", {})
        for attr_name in attrinfos.keys():
            # Check that no unexpected key appears
            for key in attrinfos[attr_name].keys():
                if key not in BaseMeta.default_attrinfo:
                    raise ValueError('_attrinfos does not handle key %s' % key)
            # Add behavior specified in attrinfo
            current_attrinfos.setdefault(attr_name, {})
            current_attrinfos[attr_name].update(attrinfos[attr_name])

        return current_attrinfos

    @staticmethod
    def inherited_classinfos(bases):
        """Looks at all classinfos dictionary of bases class and merge them to
        create the initial classinfos dictionary

        Parameters
        ----------
        bases : `list`
            All the bases of the class

        Returns
        -------
        The initial classinfos dictionary

        Notes
        -----
        index corresponds to the distance in terms of inheritance.
        The highest index (in terms of inheritance) at
        which this class has been seen. We take the highest in
        case of multiple inheritance. If we have the following :
                   A0
                  / \
                 A1 B1
                 |  |
                 A2 |
                  \/
                  A3
        We want index of A0 to be higher than index of A1 which
        inherits from A0.
        In this example:
            * A3 has index 0
            * A2 and B1 have index 1
            * A1 has index 2
            * A0 has index 3 (even if it could be 2 through B1)
        """
        classinfos = {}
        for base in bases:
            if hasattr(base, "_classinfos"):
                for cls_key in base._classinfos:
                    base_infos = base._classinfos[cls_key]

                    if cls_key in classinfos:
                        current_info = classinfos[cls_key]
                        current_info['index'] = max(current_info['index'],
                                                    base_infos['index'] + 1)
                    else:
                        classinfos[cls_key] = {}
                        classinfos[cls_key]['index'] = base_infos['index'] + 1
                        classinfos[cls_key]['attr'] = base_infos['attr']

        return classinfos

    @staticmethod
    def create_attrinfos(classinfos):
        """Browse all class in classinfos dict to create a final attrinfo dict

        Parameters
        ----------
        classinfos : `dict`
            The final classinfos dict

        Returns
        -------
        attrinfos : `dict`
            Dictionary in which key is an attribute name and value is a dict
            with all its information.
        """
        # We sort the doc reversely by index, in order to see the
        # furthest classes first (and potentially override infos of
        # parents for an attribute if two classes document it)
        attrinfos = {}
        for cls_key, info_index in sorted(classinfos.items(),
                                          key=lambda item: item[1]['index'],
                                          reverse=True):
            classinfos = info_index['attr']

            for attr_name in classinfos:
                attrinfos.setdefault(attr_name, {})
                attrinfos[attr_name].update(classinfos[attr_name])

        return attrinfos

    def __new__(mcs, class_name, bases, attrs):

        # Initialize classinfos dictionnary with all classinfos dictionnary
        # of bases
        classinfos = BaseMeta.inherited_classinfos(bases)

        # Inspect current class to have get information about its atributes
        # cls_key is an unique hashable identifier for the class
        cls_key = '%s.%s' % (attrs['__module__'], attrs['__qualname__'])
        classinfos[cls_key] = {'index': 0}
        extract_attrinfos = BaseMeta.extract_attrinfos
        classinfos[cls_key]['attr'] = extract_attrinfos(class_name, attrs)

        # Once we have collected all classinfos we can extract from it all
        # attributes information
        attrinfos = BaseMeta.create_attrinfos(classinfos)

        attrs["_classinfos"] = classinfos
        attrs["_attrinfos"] = attrinfos

        build_property = BaseMeta.build_property

        # Create properties for all attributes described in attrinfos if they
        # are not already a property; This allow us to set a special behavior
        for attr_name, info in attrinfos.items():
            attr_is_property = attrinfos[attr_name].get('is_prop', False)

            # We create the corresponding property if our item is not a property
            if not attr_is_property:
                writable = info.get("writable",
                                    BaseMeta.default_attrinfo["writable"])

                cpp_setter = info.get("cpp_setter",
                                      BaseMeta.default_attrinfo["cpp_setter"])

                attrs[attr_name] = build_property(class_name, attrs, attr_name,
                                                  writable, cpp_setter)

        # Add a __setattr__ method that forbids to add an non-existing
        # attribute
        def __setattr__(self, key, val):
            if key in attrinfos:
                object.__setattr__(self, key, val)
            else:
                raise AttributeError("'%s' object has no settable attribute "
                                     "'%s'" % (class_name, key))

        attrs["__setattr__"] = __setattr__

        # Add a method allowing to force set an attribute
        def _set(self, key: str, val):
            """A method allowing to force set an attribute
            """
            if not isinstance(key, str):
                raise ValueError(
                    'In _set function you must pass key as string')

            if key not in attrinfos:
                raise AttributeError("'%s' object has no settable attribute "
                                     "'%s'" % (class_name, key))

            object.__setattr__(self, BaseMeta.hidden_attr(key), val)

            cpp_setter = self._attrinfos[key].get(
                "cpp_setter", BaseMeta.default_attrinfo["cpp_setter"])
            if cpp_setter is not None:
                BaseMeta.set_cpp_attribute(self, val, cpp_setter)

        attrs["_set"] = _set

        return ABCMeta.__new__(mcs, class_name, bases, attrs)

    def __init__(cls, class_name, bases, attrs):
        return ABCMeta.__init__(cls, class_name, bases, attrs)


class Base(metaclass=BaseMeta):
    """The BaseClass of the tick project. This relies on some dark
    magic based on a metaclass. The aim is to have read-only attributes,
    docstring for all parameters, and some other nasty features

    Attributes
    ----------
    name : str (read-only)
        Name of the class
    """

    _attrinfos = {
        "name": {
            "writable": False
        },
    }

    def __init__(self, *args, **kwargs):
        # We add the name of the class
        self._set("name", self.__class__.__name__)

        for attr_name, prop in self.__class__.__dict__.items():
            if isinstance(prop, property):
                if attr_name in self._attrinfos and len(
                        self._attrinfos[attr_name].get('doc', [])) > 0:
                    # we create the property documentation based o what we
                    # have found in the docstring.
                    # First we will have the type of the property, then the
                    # documentation and finally the closest class (in terms
                    # of inheritance) in which it is documented
                    # Note: We join doc with '-' instead of '\n'
                    # because multiline doc does not print well in iPython

                    prop_doc = self._attrinfos[attr_name]['doc']
                    prop_doc = ' - '.join([
                        str(d).strip() for d in prop_doc
                        if len(str(d).strip()) > 0
                    ])

                    # We copy property and add the doc found in docstring
                    setattr(
                        self.__class__, attr_name,
                        property(prop.fget, prop.fset, prop.fdel, prop_doc))

    @staticmethod
    def _get_now():
        return datetime.now().strftime('%Y-%m-%d_%H-%M-%S-%f')

    def _as_dict(self):
        dd = {}
        for key in self._attrinfos.keys():
            # private and protected attributes are not shown in the
            # dict
            if not key.startswith("_"):
                dd[key] = getattr(self, key)
        return dd

    def _inc_attr(self, key: str, step: int = 1):
        """Increment an attribute of the class by ``step``

        Parameters
        ----------
        key : `str`
            Name of the class's attribute

        step : `int`
            Size of the increase
        """
        self._set(key, getattr(self, key) + step)

    def __str__(self):
        dic = self._as_dict()
        if 'dtype' in dic and isinstance(dic['dtype'], np.dtype):
            dic['dtype'] = dic['dtype'].name
        return json.dumps(dic, sort_keys=True, indent=2)
