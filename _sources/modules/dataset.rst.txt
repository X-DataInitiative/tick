
.. _dataset:

========================================
:mod:`tick.dataset`: real world datasets
========================================

Functions
---------

*tick* host real world datasets on a dedicated github repository
https://github.com/X-DataInitiative/tick-datasets .

These datasets might be every easily downloaded and cached using the
following utility function.

.. currentmodule:: tick

.. autosummary::
   :toctree: generated/
   :template: function.rst

   dataset.fetch_tick_dataset

Some datasets might also have a dedicated function handler if they need a
dedicated treatment.

.. currentmodule:: tick

.. autosummary::
   :toctree: generated/
   :template: function.rst

   dataset.fetch_hawkes_bund_data

Example
-------

.. plot:: ../examples/plot_logistic_adult.py
    :include-source:
