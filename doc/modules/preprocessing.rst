

.. _preprocessing:

=========================
:mod:`tick.preprocessing`
=========================

This module provides several preprocessing utilities, in the form of transformer
classes that change raw feature vectors into a suitable representation for some
learners. These transformers should be scikit-learn compatible, whenever possible.


Preprocessing for static features
=================================

The :class:`FeaturesBinarizer <tick.preprocessing.FeaturesBinarizer>` binarizes all
continuous features found in features matrix. This transformer is particularly
useful whenever using the :class:`ProxBinarsity <tick.prox.ProxBinarsity>`
penalization for supervised linear learning see :ref:`linear_model`.

.. currentmodule:: tick.preprocessing

.. autosummary::
   :toctree: generated/
   :template: class.rst

   FeaturesBinarizer

.. _scikit-learn preprocessing module: http://scikit-learn.org/stable/modules/preprocessing.html#preprocessing

Preprocessing for longitudinal features
=======================================

This module also provides preprocessor specific to longitudinal features with a
similar API to scikit-learn preprocessors.

.. currentmodule:: tick.preprocessing

.. autosummary::
   :toctree: generated/
   :template: class.rst

   LongitudinalFeaturesProduct
   LongitudinalFeaturesLagger
   LongitudinalSamplesFilter
