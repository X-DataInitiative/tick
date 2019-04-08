
.. _dataset:

===================
:mod:`tick.dataset`
===================

This module provides easy access to some datasets used as benchmarks in `tick`.
These datasets are hosted on the following separate repository:

    https://github.com/X-DataInitiative/tick-datasets

and are easily accessible using the following function:

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


The following datasets are easily downloadable using ``fetch_tick_dataset``
(for now, only for binary classification):

* ``binary/adult/adult.trn.bz2`` (training) and ``binary/adult/adult.tst.bz2`` (testing)
* ``binary/covtype/covtype.trn.bz2``
* ``binary/ijcnn1/ijcnn1.trn.bz2`` (training) and ``binary/ijcnn1/ijcnn1.tst.bz2`` (testing)
* ``binary/reuters/reuters.trn.bz2`` (training) and ``binary/reuters/reuters.tst.bz2`` (testing)

**Example**

.. plot:: ../examples/plot_logistic_adult.py
    :include-source:
