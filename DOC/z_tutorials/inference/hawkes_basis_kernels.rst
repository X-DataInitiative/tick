Hawkes EM with basis kernels
============================

Usage of Hawkes Basis Functions (`tick.inference.HawkesBasisKernels`).

This class is presented through experiments run on toy datasets in the
`original paper`_.

Exponential kernels
-------------------

We first attack a very basic problem in which all kernels are one exponential
functions at different scales.

.. plot:: z_tutorials/inference/code_samples/hawkes_basis_kernels_exponential.py
    :include-source:


Wave kernels: DataCos
---------------------

We also run it on a more exotic data set generated with mixtures of two cosinus
functions. We observe that we can correctly retrieve the kernels and the two
cosinus functions which have generated the kernels.

.. plot:: z_tutorials/inference/code_samples/hawkes_basis_kernels_cos.py
    :include-source:

It could have been more precise if end_time or kernel_size was increased.

.. _original paper: http://jmlr.org/proceedings/papers/v28/zhou13.html