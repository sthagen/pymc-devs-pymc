#   Copyright 2020 The PyMC Developers
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
import contextvars
import functools
import sys
import types
import warnings

from abc import ABCMeta
from functools import singledispatch
from typing import Callable, Optional, Sequence

import aesara
import numpy as np

from aesara.tensor.basic import as_tensor_variable
from aesara.tensor.random.op import RandomVariable
from aesara.tensor.random.var import RandomStateSharedVariable
from aesara.tensor.var import TensorVariable

from pymc.aesaraf import change_rv_size
from pymc.distributions import _logcdf, _logp
from pymc.distributions.shape_utils import (
    Dims,
    Shape,
    Size,
    convert_dims,
    convert_shape,
    convert_size,
    find_size,
    maybe_resize,
    resize_from_dims,
    resize_from_observed,
    to_tuple,
)
from pymc.printing import str_for_dist
from pymc.util import UNSET
from pymc.vartypes import string_types

__all__ = [
    "DensityDistRV",
    "DensityDist",
    "Distribution",
    "Continuous",
    "Discrete",
    "NoDistribution",
]

vectorized_ppc = contextvars.ContextVar(
    "vectorized_ppc", default=None
)  # type: contextvars.ContextVar[Optional[Callable]]

PLATFORM = sys.platform


class _Unpickling:
    pass


class DistributionMeta(ABCMeta):
    def __new__(cls, name, bases, clsdict):

        # Forcefully deprecate old v3 `Distribution`s
        if "random" in clsdict:

            def _random(*args, **kwargs):
                warnings.warn(
                    "The old `Distribution.random` interface is deprecated.",
                    DeprecationWarning,
                    stacklevel=2,
                )
                return clsdict["random"](*args, **kwargs)

            clsdict["random"] = _random

        rv_op = clsdict.setdefault("rv_op", None)
        rv_type = None

        if isinstance(rv_op, RandomVariable):
            rv_type = type(rv_op)

        new_cls = super().__new__(cls, name, bases, clsdict)

        if rv_type is not None:
            # Create dispatch functions

            class_logp = clsdict.get("logp")
            if class_logp:

                @_logp.register(rv_type)
                def logp(op, var, rvs_to_values, *dist_params, **kwargs):
                    value_var = rvs_to_values.get(var, var)
                    return class_logp(value_var, *dist_params, **kwargs)

            class_logcdf = clsdict.get("logcdf")
            if class_logcdf:

                @_logcdf.register(rv_type)
                def logcdf(op, var, rvs_to_values, *dist_params, **kwargs):
                    value_var = rvs_to_values.get(var, var)
                    return class_logcdf(value_var, *dist_params, **kwargs)

            class_initval = clsdict.get("get_moment")
            if class_initval:

                @_get_moment.register(rv_type)
                def get_moment(op, rv, size, *rv_inputs):
                    return class_initval(rv, size, *rv_inputs)

            # Register the Aesara `RandomVariable` type as a subclass of this
            # `Distribution` type.
            new_cls.register(rv_type)

        return new_cls


def _make_nice_attr_error(oldcode: str, newcode: str):
    def fn(*args, **kwargs):
        raise AttributeError(f"The `{oldcode}` method was removed. Instead use `{newcode}`.`")

    return fn


class Distribution(metaclass=DistributionMeta):
    """Statistical distribution"""

    rv_class = None
    rv_op = None

    def __new__(
        cls,
        name: str,
        *args,
        rng=None,
        dims: Optional[Dims] = None,
        initval=None,
        observed=None,
        total_size=None,
        transform=UNSET,
        **kwargs,
    ) -> RandomVariable:
        """Adds a RandomVariable corresponding to a PyMC distribution to the current model.

        Note that all remaining kwargs must be compatible with ``.dist()``

        Parameters
        ----------
        cls : type
            A PyMC distribution.
        name : str
            Name for the new model variable.
        rng : optional
            Random number generator to use with the RandomVariable.
        dims : tuple, optional
            A tuple of dimension names known to the model.
        initval : optional
            Test value to be attached to the output RV.
            Must match its shape exactly.
        observed : optional
            Observed data to be passed when registering the random variable in the model.
            See ``Model.register_rv``.
        total_size : float, optional
            See ``Model.register_rv``.
        transform : optional
            See ``Model.register_rv``.
        **kwargs
            Keyword arguments that will be forwarded to ``.dist()``.
            Most prominently: ``shape`` and ``size``

        Returns
        -------
        rv : RandomVariable
            The created RV, registered in the Model.
        """

        try:
            from pymc.model import Model

            model = Model.get_context()
        except TypeError:
            raise TypeError(
                "No model on context stack, which is needed to "
                "instantiate distributions. Add variable inside "
                "a 'with model:' block, or use the '.dist' syntax "
                "for a standalone distribution."
            )

        if "testval" in kwargs:
            initval = kwargs.pop("testval")
            warnings.warn(
                "The `testval` argument is deprecated; use `initval`.",
                DeprecationWarning,
                stacklevel=2,
            )

        if not isinstance(name, string_types):
            raise TypeError(f"Name needs to be a string but got: {name}")

        if rng is None:
            rng = model.next_rng()

        if dims is not None and "shape" in kwargs:
            raise ValueError(
                f"Passing both `dims` ({dims}) and `shape` ({kwargs['shape']}) is not supported!"
            )
        if dims is not None and "size" in kwargs:
            raise ValueError(
                f"Passing both `dims` ({dims}) and `size` ({kwargs['size']}) is not supported!"
            )
        dims = convert_dims(dims)

        # Create the RV without dims information, because that's not something tracked at the Aesara level.
        # If necessary we'll later replicate to a different size implied by already known dims.
        rv_out = cls.dist(*args, rng=rng, **kwargs)
        ndim_actual = rv_out.ndim
        resize_shape = None

        # `dims` are only available with this API, because `.dist()` can be used
        # without a modelcontext and dims are not tracked at the Aesara level.
        if dims is not None:
            ndim_resize, resize_shape, dims = resize_from_dims(dims, ndim_actual, model)
        elif observed is not None:
            ndim_resize, resize_shape, observed = resize_from_observed(observed, ndim_actual)

        if resize_shape:
            # A batch size was specified through `dims`, or implied by `observed`.
            rv_out = change_rv_size(rv_var=rv_out, new_size=resize_shape, expand=True)

        rv_out = model.register_rv(
            rv_out,
            name,
            observed,
            total_size,
            dims=dims,
            transform=transform,
            initval=initval,
        )

        # add in pretty-printing support
        rv_out.str_repr = types.MethodType(str_for_dist, rv_out)
        rv_out._repr_latex_ = types.MethodType(
            functools.partial(str_for_dist, formatting="latex"), rv_out
        )

        rv_out.logp = _make_nice_attr_error("rv.logp(x)", "pm.logp(rv, x)")
        rv_out.logcdf = _make_nice_attr_error("rv.logcdf(x)", "pm.logcdf(rv, x)")
        rv_out.random = _make_nice_attr_error("rv.random()", "rv.eval()")
        return rv_out

    @classmethod
    def dist(
        cls,
        dist_params,
        *,
        shape: Optional[Shape] = None,
        size: Optional[Size] = None,
        **kwargs,
    ) -> RandomVariable:
        """Creates a RandomVariable corresponding to the `cls` distribution.

        Parameters
        ----------
        dist_params : array-like
            The inputs to the `RandomVariable` `Op`.
        shape : int, tuple, Variable, optional
            A tuple of sizes for each dimension of the new RV.

            An Ellipsis (...) may be inserted in the last position to short-hand refer to
            all the dimensions that the RV would get if no shape/size/dims were passed at all.
        size : int, tuple, Variable, optional
            For creating the RV like in Aesara/NumPy.

        Returns
        -------
        rv : RandomVariable
            The created RV.
        """
        if "testval" in kwargs:
            kwargs.pop("testval")
            warnings.warn(
                "The `.dist(testval=...)` argument is deprecated and has no effect. "
                "Initial values for sampling/optimization can be specified with `initval` in a modelcontext. "
                "For using Aesara's test value features, you must assign the `.tag.test_value` yourself.",
                DeprecationWarning,
                stacklevel=2,
            )
        if "initval" in kwargs:
            raise TypeError(
                "Unexpected keyword argument `initval`. "
                "This argument is not available for the `.dist()` API."
            )

        if "dims" in kwargs:
            raise NotImplementedError("The use of a `.dist(dims=...)` API is not supported.")
        if shape is not None and size is not None:
            raise ValueError(
                f"Passing both `shape` ({shape}) and `size` ({size}) is not supported!"
            )

        shape = convert_shape(shape)
        size = convert_size(size)

        create_size, ndim_expected, ndim_batch, ndim_supp = find_size(
            shape=shape, size=size, ndim_supp=cls.rv_op.ndim_supp
        )
        # Create the RV with a `size` right away.
        # This is not necessarily the final result.
        rv_out = cls.rv_op(*dist_params, size=create_size, **kwargs)
        rv_out = maybe_resize(
            rv_out,
            cls.rv_op,
            dist_params,
            ndim_expected,
            ndim_batch,
            ndim_supp,
            shape,
            size,
            **kwargs,
        )

        rng = kwargs.pop("rng", None)
        if (
            rv_out.owner
            and isinstance(rv_out.owner.op, RandomVariable)
            and isinstance(rng, RandomStateSharedVariable)
            and not getattr(rng, "default_update", None)
        ):
            # This tells `aesara.function` that the shared RNG variable
            # is mutable, which--in turn--tells the `FunctionGraph`
            # `Supervisor` feature to allow in-place updates on the variable.
            # Without it, the `RandomVariable`s could not be optimized to allow
            # in-place RNG updates, forcing all sample results from compiled
            # functions to be the same on repeated evaluations.
            new_rng = rv_out.owner.outputs[0]
            rv_out.update = (rng, new_rng)
            rng.default_update = new_rng

        rv_out.logp = _make_nice_attr_error("rv.logp(x)", "pm.logp(rv, x)")
        rv_out.logcdf = _make_nice_attr_error("rv.logcdf(x)", "pm.logcdf(rv, x)")
        rv_out.random = _make_nice_attr_error("rv.random()", "rv.eval()")
        return rv_out


@singledispatch
def _get_moment(op, rv, size, *rv_inputs) -> TensorVariable:
    raise NotImplementedError(
        f"Random variable {rv} of type {op} has no get_moment implementation."
    )


def get_moment(rv: TensorVariable) -> TensorVariable:
    """Method for choosing a representative point/value
    that can be used to start optimization or MCMC sampling.

    The only parameter to this function is the RandomVariable
    for which the value is to be derived.
    """
    size = rv.owner.inputs[1]
    return _get_moment(rv.owner.op, rv, size, *rv.owner.inputs[3:])


class Discrete(Distribution):
    """Base class for discrete distributions"""

    def __new__(cls, name, *args, **kwargs):

        if kwargs.get("transform", None):
            raise ValueError("Transformations for discrete distributions")

        return super().__new__(cls, name, *args, **kwargs)


class Continuous(Distribution):
    """Base class for continuous distributions"""


class NoDistribution(Distribution):
    """Base class for artifical distributions

    RandomVariables that share this type are allowed in logprob graphs
    """


class DensityDistRV(RandomVariable):
    """
    Base class for DensityDistRV

    This should be subclassed when defining custom DensityDist objects.
    """

    name = "DensityDistRV"
    _print_name = ("DensityDist", "\\operatorname{DensityDist}")

    @classmethod
    def rng_fn(cls, rng, *args):
        args = list(args)
        size = args.pop(-1)
        return cls._random_fn(*args, rng=rng, size=size)


class DensityDist(NoDistribution):
    """A distribution that can be used to wrap black-box log density functions.

    Creates a Distribution and registers the supplied log density function to be used
    for inference. It is also possible to supply a `random` method in order to be able
    to sample from the prior or posterior predictive distributions.
    """

    def __new__(
        cls,
        name: str,
        *dist_params,
        logp: Optional[Callable] = None,
        logcdf: Optional[Callable] = None,
        random: Optional[Callable] = None,
        get_moment: Optional[Callable] = None,
        ndim_supp: int = 0,
        ndims_params: Optional[Sequence[int]] = None,
        dtype: str = "floatX",
        **kwargs,
    ):
        """
        Parameters
        ----------
        name : str
        dist_params : Tuple
            A sequence of the distribution's parameter. These will be converted into
            Aesara tensors internally. These parameters could be other ``RandomVariable``
            instances.
        logp : Optional[Callable]
            A callable that calculates the log density of some given observed ``value``
            conditioned on certain distribution parameter values. It must have the
            following signature: ``logp(value, *dist_params)``, where ``value`` is
            an Aesara tensor that represents the observed value, and ``dist_params``
            are the tensors that hold the values of the distribution parameters.
            This function must return an Aesara tensor. If ``None``, a ``NotImplemented``
            error will be raised when trying to compute the distribution's logp.
        logcdf : Optional[Callable]
            A callable that calculates the log cummulative probability of some given observed
            ``value`` conditioned on certain distribution parameter values. It must have the
            following signature: ``logcdf(value, *dist_params)``, where ``value`` is
            an Aesara tensor that represents the observed value, and ``dist_params``
            are the tensors that hold the values of the distribution parameters.
            This function must return an Aesara tensor. If ``None``, a ``NotImplemented``
            error will be raised when trying to compute the distribution's logcdf.
        random : Optional[Callable]
            A callable that can be used to generate random draws from the distribution.
            It must have the following signature: ``random(*dist_params, rng=None, size=None)``.
            The distribution parameters are passed as positional arguments in the
            same order as they are supplied when the ``DensityDist`` is constructed.
            The keyword arguments are ``rnd``, which will provide the random variable's
            associated :py:class:`~numpy.random.Generator`, and ``size``, that will represent
            the desired size of the random draw. If ``None``, a ``NotImplemented``
            error will be raised when trying to draw random samples from the distribution's
            prior or posterior predictive.
        get_moment : Optional[Callable]
            A callable that can be used to compute the moments of the distribution.
            It must have the following signature: ``get_moment(rv, size, *rv_inputs)``.
            The distribution's :class:`~aesara.tensor.random.op.RandomVariable` is passed
            as the first argument ``rv``. ``size`` is the random variable's size implied
            by the ``dims``, ``size`` and parameters supplied to the distribution. Finally,
            ``rv_inputs`` is the sequence of the distribution parameters, in the same order
            as they were supplied when the DensityDist was created. If ``None``, a
            ``NotImplemented`` error will be raised when trying to draw random samples from
            the distribution's prior or posterior predictive.
        ndim_supp : int
            The number of dimensions in the support of the distribution. Defaults to assuming
            a scalar distribution, i.e. ``ndim_supp = 0``.
        ndims_params : Optional[Sequence[int]]
            The list of number of dimensions in the support of each of the distribution's
            parameters. If ``None``, it is assumed that all parameters are scalars, hence
            the number of dimensions of their support will be 0.
        dtype : str
            The dtype of the distribution. All draws and observations passed into the distribution
            will be casted onto this dtype.
        kwargs :
            Extra keyword arguments are passed to the parent's class ``__new__`` method.

        Examples
        --------
            .. code-block:: python

                def logp(value, mu):
                    return -(value - mu)**2

                with pm.Model():
                    mu = pm.Normal('mu',0,1)
                    pm.DensityDist(
                        'density_dist',
                        mu,
                        logp=logp,
                        observed=np.random.randn(100),
                    )
                    idata = pm.sample(100)

            .. code-block:: python

                def logp(value, mu):
                    return -(value - mu)**2

                def random(mu, rng=None, size=None):
                    return rng.normal(loc=mu, scale=1, size=size)

                with pm.Model():
                    mu = pm.Normal('mu', 0 , 1)
                    dens = pm.DensityDist(
                        'density_dist',
                        mu,
                        logp=logp,
                        random=random,
                        observed=np.random.randn(100, 3),
                        size=(100, 3),
                    )
                    prior = pm.sample_prior_predictive(10)['density_dist']
                assert prior.shape == (10, 100, 3)

        """

        if dist_params is None:
            dist_params = []
        elif len(dist_params) > 0 and callable(dist_params[0]):
            raise TypeError(
                "The DensityDist API has changed, you are using the old API "
                "where logp was the first positional argument. In the current API, "
                "the logp is a keyword argument, amongst other changes. Please refer "
                "to the API documentation for more information on how to use the "
                "new DensityDist API."
            )
        dist_params = [as_tensor_variable(param) for param in dist_params]

        # Assume scalar ndims_params
        if ndims_params is None:
            ndims_params = [0] * len(dist_params)

        if logp is None:
            logp = default_not_implemented(name, "logp")

        if logcdf is None:
            logcdf = default_not_implemented(name, "logcdf")

        if random is None:
            random = default_not_implemented(name, "random")

        if get_moment is None:
            get_moment = default_not_implemented(name, "get_moment")

        rv_op = type(
            f"DensityDist_{name}",
            (DensityDistRV,),
            dict(
                name=f"DensityDist_{name}",
                inplace=False,
                ndim_supp=ndim_supp,
                ndims_params=ndims_params,
                dtype=dtype,
                # Specifc to DensityDist
                _random_fn=random,
            ),
        )()

        # Register custom logp
        rv_type = type(rv_op)

        @_logp.register(rv_type)
        def density_dist_logp(op, rv, rvs_to_values, *dist_params, **kwargs):
            value_var = rvs_to_values.get(rv, rv)
            return logp(
                value_var,
                *dist_params,
            )

        @_logcdf.register(rv_type)
        def density_dist_logcdf(op, var, rvs_to_values, *dist_params, **kwargs):
            value_var = rvs_to_values.get(var, var)
            return logcdf(value_var, *dist_params, **kwargs)

        @_get_moment.register(rv_type)
        def density_dist_get_moment(op, rv, size, *rv_inputs):
            return get_moment(rv, size, *rv_inputs)

        cls.rv_op = rv_op
        return super().__new__(cls, name, *dist_params, **kwargs)

    @classmethod
    def dist(cls, *args, **kwargs):
        output = super().dist(args, **kwargs)
        if cls.rv_op.dtype == "floatX":
            dtype = aesara.config.floatX
        else:
            dtype = cls.rv_op.dtype
        ndim_supp = cls.rv_op.ndim_supp
        if not hasattr(output.tag, "test_value"):
            size = to_tuple(kwargs.get("size", None)) + (1,) * ndim_supp
            output.tag.test_value = np.zeros(size, dtype)
        return output


def default_not_implemented(rv_name, method_name):
    if method_name == "random":
        # This is a hack to catch the NotImplementedError when creating the RV without random
        # If the message starts with "Cannot sample from", then it uses the test_value as
        # the initial_val.
        message = (
            f"Cannot sample from the DensityDist '{rv_name}' because the {method_name} "
            "keyword argument was not provided when the distribution was "
            f"but this method had not been provided when the distribution was "
            f"constructed. Please re-build your model and provide a callable "
            f"to '{rv_name}'s {method_name} keyword argument.\n"
        )
    else:
        message = (
            f"Attempted to run {method_name} on the DensityDist '{rv_name}', "
            f"but this method had not been provided when the distribution was "
            f"constructed. Please re-build your model and provide a callable "
            f"to '{rv_name}'s {method_name} keyword argument.\n"
        )

    def func(*args, **kwargs):
        raise NotImplementedError(message)

    return func
