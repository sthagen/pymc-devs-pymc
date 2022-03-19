***************
Transformations
***************

.. currentmodule:: pymc.distributions.transforms

Transform Instances
~~~~~~~~~~~~~~~~~~~

Transform instances are the entities that should be used in the
``transform`` parameter to a random variable constructor.

.. autosummary::
   :toctree: generated

    simplex
    logodds
    log_exp_m1
    ordered
    log
    sum_to_1
    circular


Specific Transform Classes
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated

    CholeskyCovPacked
    Interval
    LogExpM1
    Ordered
    SumTo1


Transform Composition Classes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated

    Chain
