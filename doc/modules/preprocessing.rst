

.. _preprocessing:

==================================================
:mod:`tick.preprocessing`: preprocessing utilities
==================================================

This module is an extension of the original
`scikit-learn preprocessing module`_. Just like the original one, it provides
transformer classes to change raw feature vectors into a representation that
is more suitable for estimators.

.. currentmodule:: tick

.. autosummary::
   :toctree: generated/
   :template: class.rst

   preprocessing.FeaturesBinarizer

.. _scikit-learn preprocessing module: http://scikit-learn.org/stable/modules/preprocessing.html#preprocessing

This module also provides preprocessor specific to longitudinal features with a
similar API to scikit-learn preprocessors.

.. autosummary::
   :toctree: generated/
   :template: class.rst

   preprocessing.LongitudinalFeaturesProduct
   preprocessing.LongitudinalFeaturesLagger