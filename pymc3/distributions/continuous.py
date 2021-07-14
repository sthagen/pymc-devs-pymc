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

# coding: utf-8
"""
A collection of common probability distributions for stochastic
nodes in PyMC.
"""

from typing import List, Optional, Tuple, Union

import aesara
import aesara.tensor as at
import numpy as np

from aesara.assert_op import Assert
from aesara.graph.basic import Apply
from aesara.graph.op import Op
from aesara.tensor import gammaln
from aesara.tensor.extra_ops import broadcast_shape
from aesara.tensor.random.basic import (
    BetaRV,
    WeibullRV,
    cauchy,
    chisquare,
    exponential,
    gamma,
    gumbel,
    halfcauchy,
    halfnormal,
    invgamma,
    laplace,
    logistic,
    lognormal,
    normal,
    pareto,
    triangular,
    uniform,
    vonmises,
)
from aesara.tensor.random.op import RandomVariable
from aesara.tensor.var import TensorConstant, TensorVariable

try:
    from polyagamma import polyagamma_cdf, polyagamma_pdf, random_polyagamma
except ImportError:  # pragma: no cover

    def random_polyagamma(*args, **kwargs):
        raise RuntimeError("polyagamma package is not installed!")

    def polyagamma_pdf(*args, **kwargs):
        raise RuntimeError("polyagamma package is not installed!")

    def polyagamma_cdf(*args, **kwargs):
        raise RuntimeError("polyagamma package is not installed!")


from scipy import stats
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.special import expit

from pymc3.aesaraf import floatX
from pymc3.distributions import logp_transform, transforms
from pymc3.distributions.dist_math import (
    SplineWrapper,
    betaln,
    bound,
    clipped_beta_rvs,
    i0e,
    log_i0,
    log_normal,
    logpow,
    normal_lccdf,
    normal_lcdf,
    zvalue,
)
from pymc3.distributions.distribution import Continuous
from pymc3.math import logdiffexp, logit
from pymc3.util import UNSET

__all__ = [
    "Uniform",
    "Flat",
    "HalfFlat",
    "Normal",
    "TruncatedNormal",
    "Beta",
    "Kumaraswamy",
    "Exponential",
    "Laplace",
    "StudentT",
    "Cauchy",
    "HalfCauchy",
    "Gamma",
    "Weibull",
    "HalfStudentT",
    "Lognormal",
    "ChiSquared",
    "HalfNormal",
    "Wald",
    "Pareto",
    "InverseGamma",
    "ExGaussian",
    "VonMises",
    "SkewNormal",
    "Triangular",
    "Gumbel",
    "Logistic",
    "LogitNormal",
    "Interpolated",
    "Rice",
    "Moyal",
    "AsymmetricLaplace",
    "PolyaGamma",
]


class PositiveContinuous(Continuous):
    """Base class for positive continuous distributions"""


class UnitContinuous(Continuous):
    """Base class for continuous distributions on [0,1]"""


class CircularContinuous(Continuous):
    """Base class for circular continuous distributions"""


@logp_transform.register(PositiveContinuous)
def pos_cont_transform(op):
    return transforms.log


@logp_transform.register(UnitContinuous)
def unit_cont_transform(op):
    return transforms.logodds


@logp_transform.register(CircularContinuous)
def circ_cont_transform(op):
    return transforms.circular


class BoundedContinuous(Continuous):
    """Base class for bounded continuous distributions"""

    # Indices of the arguments that define the lower and upper bounds of the distribution
    bound_args_indices = None

    def __new__(cls, *args, **kwargs):
        transform = kwargs.get("transform", UNSET)
        if transform is UNSET:
            kwargs["transform"] = cls.default_transform()
        return super().__new__(cls, *args, **kwargs)

    @classmethod
    def default_transform(cls):
        if cls.bound_args_indices is None:
            raise ValueError(
                f"Must specify bound_args_indices for {cls.__name__} bounded distribution"
            )

        def transform_params(rv_var):
            _, _, _, *args = rv_var.owner.inputs

            lower, upper = None, None
            if cls.bound_args_indices[0] is not None:
                lower = args[cls.bound_args_indices[0]]
            if cls.bound_args_indices[1] is not None:
                upper = args[cls.bound_args_indices[1]]

            if lower is not None:
                if isinstance(lower, TensorConstant) and np.all(lower.value == -np.inf):
                    lower = None
                else:
                    lower = at.as_tensor_variable(lower)

            if upper is not None:
                if isinstance(upper, TensorConstant) and np.all(upper.value == np.inf):
                    upper = None
                else:
                    upper = at.as_tensor_variable(upper)

            return lower, upper

        return transforms.interval(transform_params)


def assert_negative_support(var, label, distname, value=-1e-6):
    msg = f"The variable specified for {label} has negative support for {distname}, "
    msg += "likely making it unsuitable for this parameter."
    return Assert(msg)(var, at.all(at.ge(var, 0.0)))


def get_tau_sigma(tau=None, sigma=None):
    r"""
    Find precision and standard deviation. The link between the two
    parameterizations is given by the inverse relationship:

    .. math::
        \tau = \frac{1}{\sigma^2}

    Parameters
    ----------
    tau: array-like, optional
    sigma: array-like, optional

    Results
    -------
    Returns tuple (tau, sigma)

    Notes
    -----
    If neither tau nor sigma is provided, returns (1., 1.)
    """
    if tau is None:
        if sigma is None:
            sigma = 1.0
            tau = 1.0
        else:
            tau = sigma ** -2.0

    else:
        if sigma is not None:
            raise ValueError("Can't pass both tau and sigma")
        else:
            sigma = tau ** -0.5

    return floatX(tau), floatX(sigma)


class Uniform(BoundedContinuous):
    r"""
    Continuous uniform log-likelihood.

    The pdf of this distribution is

    .. math::

       f(x \mid lower, upper) = \frac{1}{upper-lower}

    .. plot::

        import matplotlib.pyplot as plt
        import numpy as np
        import arviz as az
        plt.style.use('arviz-darkgrid')
        x = np.linspace(-3, 3, 500)
        ls = [0., -2]
        us = [2., 1]
        for l, u in zip(ls, us):
            y = np.zeros(500)
            y[(x<u) & (x>l)] = 1.0/(u-l)
            plt.plot(x, y, label='lower = {}, upper = {}'.format(l, u))
        plt.xlabel('x', fontsize=12)
        plt.ylabel('f(x)', fontsize=12)
        plt.ylim(0, 1)
        plt.legend(loc=1)
        plt.show()

    ========  =====================================
    Support   :math:`x \in [lower, upper]`
    Mean      :math:`\dfrac{lower + upper}{2}`
    Variance  :math:`\dfrac{(upper - lower)^2}{12}`
    ========  =====================================

    Parameters
    ----------
    lower: float
        Lower limit.
    upper: float
        Upper limit.
    """
    rv_op = uniform
    bound_args_indices = (0, 1)  # Lower, Upper

    @classmethod
    def dist(cls, lower=0, upper=1, **kwargs):
        lower = at.as_tensor_variable(floatX(lower))
        upper = at.as_tensor_variable(floatX(upper))
        return super().dist([lower, upper], **kwargs)

    def logp(value, lower, upper):
        """
        Calculate log-probability of Uniform distribution at specified value.

        Parameters
        ----------
        value: numeric
            Value for which log-probability is calculated.

        Returns
        -------
        TensorVariable
        """
        return bound(
            at.fill(value, -at.log(upper - lower)),
            value >= lower,
            value <= upper,
        )

    def logcdf(value, lower, upper):
        """
        Compute the log of the cumulative distribution function for Uniform distribution
        at the specified value.

        Parameters
        ----------
        value: numeric or np.ndarray or `TensorVariable`
            Value(s) for which log CDF is calculated. If the log CDF for multiple
            values are desired the values must be provided in a numpy array or `TensorVariable`.

        Returns
        -------
        TensorVariable
        """
        return at.switch(
            at.lt(value, lower) | at.lt(upper, lower),
            -np.inf,
            at.switch(
                at.lt(value, upper),
                at.log(value - lower) - at.log(upper - lower),
                0,
            ),
        )


class FlatRV(RandomVariable):
    name = "flat"
    ndim_supp = 0
    ndims_params = []
    dtype = "floatX"
    _print_name = ("Flat", "\\operatorname{Flat}")

    @classmethod
    def rng_fn(cls, rng, size):
        raise NotImplementedError("Cannot sample from flat variable")


flat = FlatRV()


class Flat(Continuous):
    """
    Uninformative log-likelihood that returns 0 regardless of
    the passed value.
    """

    rv_op = flat

    @classmethod
    def dist(cls, *, size=None, initval=None, **kwargs):
        if initval is None:
            initval = np.full(size, floatX(0.0))
        res = super().dist([], size=size, **kwargs)
        res.tag.test_value = initval
        return res

    def logp(value):
        """
        Calculate log-probability of Flat distribution at specified value.

        Parameters
        ----------
        value: numeric
            Value(s) for which log-probability is calculated. If the log probabilities for multiple
            values are desired the values must be provided in a numpy array or Aesara tensor

        Returns
        -------
        TensorVariable
        """
        return at.zeros_like(value)

    def logcdf(value):
        """
        Compute the log of the cumulative distribution function for Flat distribution
        at the specified value.

        Parameters
        ----------
        value: numeric or np.ndarray or aesara.tensor
            Value(s) for which log CDF is calculated. If the log CDF for multiple
            values are desired the values must be provided in a numpy array or Aesara tensor.

        Returns
        -------
        TensorVariable
        """
        return at.switch(
            at.eq(value, -np.inf), -np.inf, at.switch(at.eq(value, np.inf), 0, at.log(0.5))
        )


class HalfFlatRV(RandomVariable):
    name = "half_flat"
    ndim_supp = 0
    ndims_params = []
    dtype = "floatX"
    _print_name = ("HalfFlat", "\\operatorname{HalfFlat}")

    @classmethod
    def rng_fn(cls, rng, size):
        raise NotImplementedError("Cannot sample from half_flat variable")


halfflat = HalfFlatRV()


class HalfFlat(PositiveContinuous):
    """Improper flat prior over the positive reals."""

    rv_op = halfflat

    @classmethod
    def dist(cls, *, size=None, initval=None, **kwargs):
        if initval is None:
            initval = np.full(size, floatX(1.0))
        res = super().dist([], size=size, **kwargs)
        res.tag.test_value = initval
        return res

    def logp(value):
        """
        Calculate log-probability of HalfFlat distribution at specified value.

        Parameters
        ----------
        value: numeric
            Value(s) for which log-probability is calculated. If the log probabilities for multiple
            values are desired the values must be provided in a numpy array or Aesara tensor

        Returns
        -------
        TensorVariable
        """
        return bound(at.zeros_like(value), value > 0)

    def logcdf(value):
        """
        Compute the log of the cumulative distribution function for HalfFlat distribution
        at the specified value.

        Parameters
        ----------
        value: numeric or np.ndarray or aesara.tensor
            Value(s) for which log CDF is calculated. If the log CDF for multiple
            values are desired the values must be provided in a numpy array or Aesara tensor.

        Returns
        -------
        TensorVariable
        """
        return at.switch(at.lt(value, np.inf), -np.inf, at.switch(at.eq(value, np.inf), 0, -np.inf))


class Normal(Continuous):
    r"""
    Univariate normal log-likelihood.

    The pdf of this distribution is

    .. math::

       f(x \mid \mu, \tau) =
           \sqrt{\frac{\tau}{2\pi}}
           \exp\left\{ -\frac{\tau}{2} (x-\mu)^2 \right\}

    Normal distribution can be parameterized either in terms of precision
    or standard deviation. The link between the two parametrizations is
    given by

    .. math::

       \tau = \dfrac{1}{\sigma^2}

    .. plot::

        import matplotlib.pyplot as plt
        import numpy as np
        import scipy.stats as st
        import arviz as az
        plt.style.use('arviz-darkgrid')
        x = np.linspace(-5, 5, 1000)
        mus = [0., 0., 0., -2.]
        sigmas = [0.4, 1., 2., 0.4]
        for mu, sigma in zip(mus, sigmas):
            pdf = st.norm.pdf(x, mu, sigma)
            plt.plot(x, pdf, label=r'$\mu$ = {}, $\sigma$ = {}'.format(mu, sigma))
        plt.xlabel('x', fontsize=12)
        plt.ylabel('f(x)', fontsize=12)
        plt.legend(loc=1)
        plt.show()

    ========  ==========================================
    Support   :math:`x \in \mathbb{R}`
    Mean      :math:`\mu`
    Variance  :math:`\dfrac{1}{\tau}` or :math:`\sigma^2`
    ========  ==========================================

    Parameters
    ----------
    mu: float
        Mean.
    sigma: float
        Standard deviation (sigma > 0) (only required if tau is not specified).
    tau: float
        Precision (tau > 0) (only required if sigma is not specified).

    Examples
    --------
    .. code-block:: python

        with pm.Model():
            x = pm.Normal('x', mu=0, sigma=10)

        with pm.Model():
            x = pm.Normal('x', mu=0, tau=1/23)
    """
    rv_op = normal

    @classmethod
    def dist(cls, mu=0, sigma=None, tau=None, sd=None, no_assert=False, **kwargs):
        if sd is not None:
            sigma = sd
        tau, sigma = get_tau_sigma(tau=tau, sigma=sigma)
        sigma = at.as_tensor_variable(sigma)

        # sd = sigma
        # tau = at.as_tensor_variable(tau)
        # mean = median = mode = mu = at.as_tensor_variable(floatX(mu))
        # variance = 1.0 / self.tau

        if not no_assert:
            assert_negative_support(sigma, "sigma", "Normal")

        return super().dist([mu, sigma], **kwargs)

    def logp(value, mu, sigma):
        """
        Calculate log-probability of Normal distribution at specified value.

        Parameters
        ----------
        value: numeric
            Value(s) for which log-probability is calculated. If the log probabilities for multiple
            values are desired the values must be provided in a numpy array or `TensorVariable`.

        Returns
        -------
        TensorVariable
        """
        tau, sigma = get_tau_sigma(tau=None, sigma=sigma)

        return bound((-tau * (value - mu) ** 2 + at.log(tau / np.pi / 2.0)) / 2.0, sigma > 0)

    def logcdf(value, mu, sigma):
        """
        Compute the log of the cumulative distribution function for Normal distribution
        at the specified value.

        Parameters
        ----------
        value: numeric or np.ndarray or `TensorVariable`
            Value(s) for which log CDF is calculated. If the log CDF for multiple
            values are desired the values must be provided in a numpy array or `TensorVariable`.

        Returns
        -------
        TensorVariable
        """
        return bound(
            normal_lcdf(mu, sigma, value),
            0 < sigma,
        )


class TruncatedNormalRV(RandomVariable):
    name = "truncated_normal"
    ndim_supp = 0
    ndims_params = [0, 0, 0, 0]
    dtype = "floatX"
    _print_name = ("TruncatedNormal", "\\operatorname{TruncatedNormal}")

    @classmethod
    def rng_fn(
        cls,
        rng: np.random.RandomState,
        mu: Union[np.ndarray, float],
        sigma: Union[np.ndarray, float],
        lower: Union[np.ndarray, float],
        upper: Union[np.ndarray, float],
        size: Optional[Union[List[int], int]],
    ) -> np.ndarray:
        return stats.truncnorm.rvs(
            a=(lower - mu) / sigma,
            b=(upper - mu) / sigma,
            loc=mu,
            scale=sigma,
            size=size,
            random_state=rng,
        )


truncated_normal = TruncatedNormalRV()


class TruncatedNormal(BoundedContinuous):
    r"""
    Univariate truncated normal log-likelihood.

    The pdf of this distribution is

    .. math::

       f(x;\mu ,\sigma ,a,b)={\frac {\phi ({\frac {x-\mu }{\sigma }})}{
       \sigma \left(\Phi ({\frac {b-\mu }{\sigma }})-\Phi ({\frac {a-\mu }{\sigma }})\right)}}

    Truncated normal distribution can be parameterized either in terms of precision
    or standard deviation. The link between the two parametrizations is
    given by

    .. math::

       \tau = \dfrac{1}{\sigma^2}


    .. plot::

        import matplotlib.pyplot as plt
        import numpy as np
        import scipy.stats as st
        import arviz as az
        plt.style.use('arviz-darkgrid')
        x = np.linspace(-10, 10, 1000)
        mus = [0.,  0., 0.]
        sigmas = [3.,5.,7.]
        a1 = [-3, -5, -5]
        b1 = [7, 5, 4]
        for mu, sigma, a, b in zip(mus, sigmas,a1,b1):
            an, bn = (a - mu) / sigma, (b - mu) / sigma
            pdf = st.truncnorm.pdf(x, an,bn, loc=mu, scale=sigma)
            plt.plot(x, pdf, label=r'$\mu$ = {}, $\sigma$ = {}, a={}, b={}'.format(mu, sigma, a, b))
        plt.xlabel('x', fontsize=12)
        plt.ylabel('f(x)', fontsize=12)
        plt.legend(loc=1)
        plt.show()

    ========  ==========================================
    Support   :math:`x \in [a, b]`
    Mean      :math:`\mu +{\frac {\phi (\alpha )-\phi (\beta )}{Z}}\sigma`
    Variance  :math:`\sigma ^{2}\left[1+{\frac {\alpha \phi (\alpha )-\beta \phi (\beta )}{Z}}-\left({\frac {\phi (\alpha )-\phi (\beta )}{Z}}\right)^{2}\right]`
    ========  ==========================================

    Parameters
    ----------
    mu: float
        Mean.
    sigma: float
        Standard deviation (sigma > 0).
    lower: float (optional)
        Left bound.
    upper: float (optional)
        Right bound.

    Examples
    --------
    .. code-block:: python

        with pm.Model():
            x = pm.TruncatedNormal('x', mu=0, sigma=10, lower=0)

        with pm.Model():
            x = pm.TruncatedNormal('x', mu=0, sigma=10, upper=1)

        with pm.Model():
            x = pm.TruncatedNormal('x', mu=0, sigma=10, lower=0, upper=1)

    """

    rv_op = truncated_normal
    bound_args_indices = (2, 3)  # indexes for lower and upper args

    @classmethod
    def dist(
        cls,
        mu: Optional[Union[float, np.ndarray]] = None,
        sigma: Optional[Union[float, np.ndarray]] = None,
        tau: Optional[Union[float, np.ndarray]] = None,
        sd: Optional[Union[float, np.ndarray]] = None,
        lower: Optional[Union[float, np.ndarray]] = None,
        upper: Optional[Union[float, np.ndarray]] = None,
        transform: str = "auto",
        *args,
        **kwargs,
    ) -> RandomVariable:
        sigma = sd if sd is not None else sigma
        tau, sigma = get_tau_sigma(tau=tau, sigma=sigma)
        sigma = at.as_tensor_variable(sigma)
        tau = at.as_tensor_variable(tau)
        mu = at.as_tensor_variable(floatX(mu))
        assert_negative_support(sigma, "sigma", "TruncatedNormal")
        assert_negative_support(tau, "tau", "TruncatedNormal")

        # if lower is None and upper is None:
        #     initval = mu
        # elif lower is None and upper is not None:
        #     initval = upper - 1.0
        # elif lower is not None and upper is None:
        #     initval = lower + 1.0
        # else:
        #     initval = (lower + upper) / 2

        lower = at.as_tensor_variable(floatX(lower)) if lower is not None else at.constant(-np.inf)
        upper = at.as_tensor_variable(floatX(upper)) if upper is not None else at.constant(np.inf)
        return super().dist([mu, sigma, lower, upper], **kwargs)

    def logp(
        value,
        mu: Union[float, np.ndarray, TensorVariable],
        sigma: Union[float, np.ndarray, TensorVariable],
        lower: Union[float, np.ndarray, TensorVariable],
        upper: Union[float, np.ndarray, TensorVariable],
    ) -> RandomVariable:
        """
        Calculate log-probability of TruncatedNormal distribution at specified value.

        Parameters
        ----------
        value: numeric
            Value(s) for which log-probability is calculated. If the log probabilities for multiple
            values are desired the values must be provided in a numpy array or Aesara tensor

        Returns
        -------
        TensorVariable
        """
        unbounded_lower = isinstance(lower, TensorConstant) and np.all(lower.value == -np.inf)
        unbounded_upper = isinstance(upper, TensorConstant) and np.all(upper.value == np.inf)

        if not unbounded_lower and not unbounded_upper:
            lcdf_a = normal_lcdf(mu, sigma, lower)
            lcdf_b = normal_lcdf(mu, sigma, upper)
            lsf_a = normal_lccdf(mu, sigma, lower)
            lsf_b = normal_lccdf(mu, sigma, upper)
            norm = at.switch(lower > 0, logdiffexp(lsf_a, lsf_b), logdiffexp(lcdf_b, lcdf_a))
        elif not unbounded_lower:
            norm = normal_lccdf(mu, sigma, lower)
        elif not unbounded_upper:
            norm = normal_lcdf(mu, sigma, upper)
        else:
            norm = 0.0

        logp = Normal.logp(value, mu=mu, sigma=sigma) - norm
        bounds = []
        if not unbounded_lower:
            bounds.append(value >= lower)
        if not unbounded_upper:
            bounds.append(value <= upper)
        if not unbounded_lower and not unbounded_upper:
            bounds.append(lower <= upper)
        return bound(logp, *bounds)


class HalfNormal(PositiveContinuous):
    r"""
    Half-normal log-likelihood.

    The pdf of this distribution is

    .. math::

       f(x \mid \tau) =
           \sqrt{\frac{2\tau}{\pi}}
           \exp\left(\frac{-x^2 \tau}{2}\right)

       f(x \mid \sigma) =
           \sqrt{\frac{2}{\pi\sigma^2}}
           \exp\left(\frac{-x^2}{2\sigma^2}\right)

    .. note::

       The parameters ``sigma``/``tau`` (:math:`\sigma`/:math:`\tau`) refer to
       the standard deviation/precision of the unfolded normal distribution, for
       the standard deviation of the half-normal distribution, see below. For
       the half-normal, they are just two parameterisation :math:`\sigma^2
       \equiv \frac{1}{\tau}` of a scale parameter

    .. plot::

        import matplotlib.pyplot as plt
        import numpy as np
        import scipy.stats as st
        import arviz as az
        plt.style.use('arviz-darkgrid')
        x = np.linspace(0, 5, 200)
        for sigma in [0.4, 1., 2.]:
            pdf = st.halfnorm.pdf(x, scale=sigma)
            plt.plot(x, pdf, label=r'$\sigma$ = {}'.format(sigma))
        plt.xlabel('x', fontsize=12)
        plt.ylabel('f(x)', fontsize=12)
        plt.legend(loc=1)
        plt.show()

    ========  ==========================================
    Support   :math:`x \in [0, \infty)`
    Mean      :math:`\sqrt{\dfrac{2}{\tau \pi}}` or :math:`\dfrac{\sigma \sqrt{2}}{\sqrt{\pi}}`
    Variance  :math:`\dfrac{1}{\tau}\left(1 - \dfrac{2}{\pi}\right)` or :math:`\sigma^2\left(1 - \dfrac{2}{\pi}\right)`
    ========  ==========================================

    Parameters
    ----------
    sigma: float
        Scale parameter :math:`sigma` (``sigma`` > 0) (only required if ``tau`` is not specified).
    tau: float
        Precision :math:`tau` (tau > 0) (only required if sigma is not specified).

    Examples
    --------
    .. code-block:: python

        with pm.Model():
            x = pm.HalfNormal('x', sigma=10)

        with pm.Model():
            x = pm.HalfNormal('x', tau=1/15)
    """
    rv_op = halfnormal

    @classmethod
    def dist(cls, sigma=None, tau=None, sd=None, *args, **kwargs):
        if sd is not None:
            sigma = sd

        tau, sigma = get_tau_sigma(tau=tau, sigma=sigma)

        assert_negative_support(tau, "tau", "HalfNormal")
        assert_negative_support(sigma, "sigma", "HalfNormal")

        return super().dist([0.0, sigma], **kwargs)

    def logp(value, loc, sigma):
        """
        Calculate log-probability of HalfNormal distribution at specified value.

        Parameters
        ----------
        value: numeric
            Value(s) for which log-probability is calculated. If the log probabilities for multiple
            values are desired the values must be provided in a numpy array or Aesara tensor

        Returns
        -------
        TensorVariable
        """
        tau, sigma = get_tau_sigma(tau=None, sigma=sigma)

        return bound(
            -0.5 * tau * (value - loc) ** 2 + 0.5 * at.log(tau * 2.0 / np.pi),
            value >= loc,
            tau > 0,
            sigma > 0,
        )

    def logcdf(value, loc, sigma):
        """
        Compute the log of the cumulative distribution function for HalfNormal distribution
        at the specified value.

        Parameters
        ----------
        value: numeric or np.ndarray or aesara.tensor
            Value(s) for which log CDF is calculated. If the log CDF for multiple
            values are desired the values must be provided in a numpy array or Aesara tensor.

        Returns
        -------
        TensorVariable
        """
        z = zvalue(value, mu=loc, sigma=sigma)
        return bound(
            at.log1p(-at.erfc(z / at.sqrt(2.0))),
            loc <= value,
            0 < sigma,
        )

    def _distr_parameters_for_repr(self):
        return ["sigma"]


class WaldRV(RandomVariable):
    name = "wald"
    ndim_supp = 0
    ndims_params = [0, 0, 0]
    dtype = "floatX"
    _print_name = ("Wald", "\\operatorname{Wald}")

    @classmethod
    def rng_fn(cls, rng, mu, lam, alpha, size):
        return rng.wald(mu, lam, size=size) + alpha


wald = WaldRV()


class Wald(PositiveContinuous):
    r"""
    Wald log-likelihood.

    The pdf of this distribution is

    .. math::

       f(x \mid \mu, \lambda) =
           \left(\frac{\lambda}{2\pi}\right)^{1/2} x^{-3/2}
           \exp\left\{
               -\frac{\lambda}{2x}\left(\frac{x-\mu}{\mu}\right)^2
           \right\}

    .. plot::

        import matplotlib.pyplot as plt
        import numpy as np
        import scipy.stats as st
        import arviz as az
        plt.style.use('arviz-darkgrid')
        x = np.linspace(0, 3, 500)
        mus = [1., 1., 1., 3.]
        lams = [1., .2, 3., 1.]
        for mu, lam in zip(mus, lams):
            pdf = st.invgauss.pdf(x, mu/lam, scale=lam)
            plt.plot(x, pdf, label=r'$\mu$ = {}, $\lambda$ = {}'.format(mu, lam))
        plt.xlabel('x', fontsize=12)
        plt.ylabel('f(x)', fontsize=12)
        plt.legend(loc=1)
        plt.show()

    ========  =============================
    Support   :math:`x \in (0, \infty)`
    Mean      :math:`\mu`
    Variance  :math:`\dfrac{\mu^3}{\lambda}`
    ========  =============================

    Wald distribution can be parameterized either in terms of lam or phi.
    The link between the two parametrizations is given by

    .. math::

       \phi = \dfrac{\lambda}{\mu}

    Parameters
    ----------
    mu: float, optional
        Mean of the distribution (mu > 0).
    lam: float, optional
        Relative precision (lam > 0).
    phi: float, optional
        Alternative shape parameter (phi > 0).
    alpha: float, optional
        Shift/location parameter (alpha >= 0).

    Notes
    -----
    To instantiate the distribution specify any of the following

    - only mu (in this case lam will be 1)
    - mu and lam
    - mu and phi
    - lam and phi

    References
    ----------
    .. [Tweedie1957] Tweedie, M. C. K. (1957).
       Statistical Properties of Inverse Gaussian Distributions I.
       The Annals of Mathematical Statistics, Vol. 28, No. 2, pp. 362-377

    .. [Michael1976] Michael, J. R., Schucany, W. R. and Hass, R. W. (1976).
        Generating Random Variates Using Transformations with Multiple Roots.
        The American Statistician, Vol. 30, No. 2, pp. 88-90

    .. [Giner2016] Göknur Giner, Gordon K. Smyth (2016)
       statmod: Probability Calculations for the Inverse Gaussian Distribution
    """
    rv_op = wald

    @classmethod
    def dist(
        cls,
        mu: Optional[Union[float, np.ndarray]] = None,
        lam: Optional[Union[float, np.ndarray]] = None,
        phi: Optional[Union[float, np.ndarray]] = None,
        alpha: Union[float, np.ndarray] = 0.0,
        *args,
        **kwargs,
    ) -> RandomVariable:
        mu, lam, phi = cls.get_mu_lam_phi(mu, lam, phi)
        alpha = at.as_tensor_variable(floatX(alpha))
        mu = at.as_tensor_variable(floatX(mu))
        lam = at.as_tensor_variable(floatX(lam))

        assert_negative_support(phi, "phi", "Wald")
        assert_negative_support(mu, "mu", "Wald")
        assert_negative_support(lam, "lam", "Wald")

        return super().dist([mu, lam, alpha], **kwargs)

    @staticmethod
    def get_mu_lam_phi(
        mu: Optional[float], lam: Optional[float], phi: Optional[float]
    ) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray], Union[float, np.ndarray]]:
        if mu is None:
            if lam is not None and phi is not None:
                return lam / phi, lam, phi
        else:
            if lam is None:
                if phi is None:
                    return mu, 1.0, 1.0 / mu
                else:
                    return mu, mu * phi, phi
            else:
                if phi is None:
                    return mu, lam, lam / mu

        raise ValueError(
            "Wald distribution must specify either mu only, "
            "mu and lam, mu and phi, or lam and phi."
        )

    def logp(
        value,
        mu: Union[float, np.ndarray, TensorVariable],
        lam: Union[float, np.ndarray, TensorVariable],
        alpha: Union[float, np.ndarray, TensorVariable],
    ) -> RandomVariable:
        """
        Calculate log-probability of Wald distribution at specified value.

        Parameters
        ----------
        value: numeric
            Value(s) for which log-probability is calculated. If the log probabilities for multiple
            values are desired the values must be provided in a numpy array or Aesara tensor
        mu: float or TensorVariable
            Mean of the distribution (mu > 0).
        lam: float or TensorVariable
            Relative precision (lam > 0).
        alpha: float or TensorVariable
            Shift/location parameter (alpha >= 0).

        Returns
        -------
        TensorVariable
        """
        centered_value = value - alpha
        # value *must* be iid. Otherwise this is wrong.
        return bound(
            logpow(lam / (2.0 * np.pi), 0.5)
            - logpow(centered_value, 1.5)
            - (0.5 * lam / centered_value * ((centered_value - mu) / mu) ** 2),
            centered_value > 0,
            mu > 0,
            lam > 0,
            alpha >= 0,
        )

    def _distr_parameters_for_repr(self):
        return ["mu", "lam", "alpha"]

    def logcdf(
        value,
        mu: Union[float, np.ndarray, TensorVariable],
        lam: Union[float, np.ndarray, TensorVariable],
        alpha: Union[float, np.ndarray, TensorVariable],
    ) -> RandomVariable:
        """
        Compute the log of the cumulative distribution function for Wald distribution
        at the specified value.

        Parameters
        ----------
        value: numeric or np.ndarray or aesara.tensor
            Value(s) for which log CDF is calculated. If the log CDF for multiple
            values are desired the values must be provided in a numpy array or Aesara tensor.
        mu: float or TensorVariable
            Mean of the distribution (mu > 0).
        lam: float or TensorVariable
            Relative precision (lam > 0).
        alpha: float or TensorVariable
            Shift/location parameter (alpha >= 0).

        Returns
        -------
        TensorVariable
        """
        value -= alpha
        q = value / mu
        l = lam * mu
        r = at.sqrt(value * lam)

        a = normal_lcdf(0, 1, (q - 1.0) / r)
        b = 2.0 / l + normal_lcdf(0, 1, -(q + 1.0) / r)

        return bound(
            at.switch(
                at.lt(value, np.inf),
                a + at.log1pexp(b - a),
                0,
            ),
            0 < value,
            0 < mu,
            0 < lam,
            0 <= alpha,
        )


class BetaClippedRV(BetaRV):
    @classmethod
    def rng_fn(cls, rng, alpha, beta, size):
        return clipped_beta_rvs(alpha, beta, size=size, random_state=rng)


beta = BetaClippedRV()


class Beta(UnitContinuous):
    r"""
    Beta log-likelihood.

    The pdf of this distribution is

    .. math::

       f(x \mid \alpha, \beta) =
           \frac{x^{\alpha - 1} (1 - x)^{\beta - 1}}{B(\alpha, \beta)}

    .. plot::

        import matplotlib.pyplot as plt
        import numpy as np
        import scipy.stats as st
        import arviz as az
        plt.style.use('arviz-darkgrid')
        x = np.linspace(0, 1, 200)
        alphas = [.5, 5., 1., 2., 2.]
        betas = [.5, 1., 3., 2., 5.]
        for a, b in zip(alphas, betas):
            pdf = st.beta.pdf(x, a, b)
            plt.plot(x, pdf, label=r'$\alpha$ = {}, $\beta$ = {}'.format(a, b))
        plt.xlabel('x', fontsize=12)
        plt.ylabel('f(x)', fontsize=12)
        plt.ylim(0, 4.5)
        plt.legend(loc=9)
        plt.show()

    ========  ==============================================================
    Support   :math:`x \in (0, 1)`
    Mean      :math:`\dfrac{\alpha}{\alpha + \beta}`
    Variance  :math:`\dfrac{\alpha \beta}{(\alpha+\beta)^2(\alpha+\beta+1)}`
    ========  ==============================================================

    Beta distribution can be parameterized either in terms of alpha and
    beta or mean and standard deviation. The link between the two
    parametrizations is given by

    .. math::

       \alpha &= \mu \kappa \\
       \beta  &= (1 - \mu) \kappa

       \text{where } \kappa = \frac{\mu(1-\mu)}{\sigma^2} - 1

    Parameters
    ----------
    alpha: float
        alpha > 0.
    beta: float
        beta > 0.
    mu: float
        Alternative mean (0 < mu < 1).
    sigma: float
        Alternative standard deviation (0 < sigma < sqrt(mu * (1 - mu))).

    Notes
    -----
    Beta distribution is a conjugate prior for the parameter :math:`p` of
    the binomial distribution.
    """

    rv_op = beta

    @classmethod
    def dist(cls, alpha=None, beta=None, mu=None, sigma=None, sd=None, *args, **kwargs):
        if sd is not None:
            sigma = sd

        alpha, beta = cls.get_alpha_beta(alpha, beta, mu, sigma)
        alpha = at.as_tensor_variable(floatX(alpha))
        beta = at.as_tensor_variable(floatX(beta))

        assert_negative_support(alpha, "alpha", "Beta")
        assert_negative_support(beta, "beta", "Beta")

        return super().dist([alpha, beta], **kwargs)

    @classmethod
    def get_alpha_beta(self, alpha=None, beta=None, mu=None, sigma=None):
        if (alpha is not None) and (beta is not None):
            pass
        elif (mu is not None) and (sigma is not None):
            kappa = mu * (1 - mu) / sigma ** 2 - 1
            alpha = mu * kappa
            beta = (1 - mu) * kappa
        else:
            raise ValueError(
                "Incompatible parameterization. Either use alpha "
                "and beta, or mu and sigma to specify distribution."
            )

        return alpha, beta

    def _distr_parameters_for_repr(self):
        return ["alpha", "beta"]

    def logp(value, alpha, beta):
        """
        Calculate log-probability of Beta distribution at specified value.

        Parameters
        ----------
        value: numeric
            Value(s) for which log-probability is calculated. If the log probabilities for multiple
            values are desired the values must be provided in a numpy array or Aesara tensor

        Returns
        -------
        TensorVariable
        """

        logval = at.log(value)
        log1pval = at.log1p(-value)
        logp = (
            at.switch(at.eq(alpha, 1), 0, (alpha - 1) * logval)
            + at.switch(at.eq(beta, 1), 0, (beta - 1) * log1pval)
            - betaln(alpha, beta)
        )

        return bound(logp, value >= 0, value <= 1, alpha > 0, beta > 0)

    def logcdf(value, alpha, beta):
        """
        Compute the log of the cumulative distribution function for Beta distribution
        at the specified value.

        Parameters
        ----------
        value: numeric or np.ndarray or aesara.tensor
            Value(s) for which log CDF is calculated. If the log CDF for multiple
            values are desired the values must be provided in a numpy array or Aesara tensor.

        Returns
        -------
        TensorVariable
        """

        return bound(
            at.switch(
                at.lt(value, 1),
                at.log(at.betainc(alpha, beta, value)),
                0,
            ),
            0 <= value,
            0 < alpha,
            0 < beta,
        )


class KumaraswamyRV(RandomVariable):
    name = "kumaraswamy"
    ndim_supp = 0
    ndims_params = [0, 0]
    dtype = "floatX"
    _print_name = ("Kumaraswamy", "\\operatorname{Kumaraswamy}")

    @classmethod
    def rng_fn(cls, rng, a, b, size):
        u = rng.uniform(size=size)
        return (1 - (1 - u) ** (1 / b)) ** (1 / a)


kumaraswamy = KumaraswamyRV()


class Kumaraswamy(UnitContinuous):
    r"""
    Kumaraswamy log-likelihood.

    The pdf of this distribution is

    .. math::

       f(x \mid a, b) =
           abx^{a-1}(1-x^a)^{b-1}

    .. plot::

        import matplotlib.pyplot as plt
        import numpy as np
        import arviz as az
        plt.style.use('arviz-darkgrid')
        x = np.linspace(0, 1, 200)
        a_s = [.5, 5., 1., 2., 2.]
        b_s = [.5, 1., 3., 2., 5.]
        for a, b in zip(a_s, b_s):
            pdf = a * b * x ** (a - 1) * (1 - x ** a) ** (b - 1)
            plt.plot(x, pdf, label=r'$a$ = {}, $b$ = {}'.format(a, b))
        plt.xlabel('x', fontsize=12)
        plt.ylabel('f(x)', fontsize=12)
        plt.ylim(0, 3.)
        plt.legend(loc=9)
        plt.show()

    ========  ==============================================================
    Support   :math:`x \in (0, 1)`
    Mean      :math:`b B(1 + \tfrac{1}{a}, b)`
    Variance  :math:`b B(1 + \tfrac{2}{a}, b) - (b B(1 + \tfrac{1}{a}, b))^2`
    ========  ==============================================================

    Parameters
    ----------
    a: float
        a > 0.
    b: float
        b > 0.
    """
    rv_op = kumaraswamy

    @classmethod
    def dist(cls, a, b, *args, **kwargs):
        a = at.as_tensor_variable(floatX(a))
        b = at.as_tensor_variable(floatX(b))

        assert_negative_support(a, "a", "Kumaraswamy")
        assert_negative_support(b, "b", "Kumaraswamy")

        return super().dist([a, b], *args, **kwargs)

    def logp(value, a, b):
        """
        Calculate log-probability of Kumaraswamy distribution at specified value.

        Parameters
        ----------
        value: numeric
            Value(s) for which log-probability is calculated. If the log probabilities for multiple
            values are desired the values must be provided in a numpy array or Aesara tensor

        Returns
        -------
        TensorVariable
        """
        logp = at.log(a) + at.log(b) + (a - 1) * at.log(value) + (b - 1) * at.log(1 - value ** a)

        return bound(logp, value >= 0, value <= 1, a > 0, b > 0)

    def logcdf(value, a, b):
        r"""
        Compute the log of cumulative distribution function for the Kumaraswamy distribution
        at the specified value.

        Parameters
        ----------
        value: numeric or np.ndarray or aesara.tensor
            Value(s) for which log CDF is calculated. If the log CDF for
            multiple values are desired the values must be provided in a numpy
            array or Aesara tensor.

        Returns
        -------
        TensorVariable
        """
        logcdf = at.log1mexp(b * at.log1p(-(value ** a)))
        return bound(at.switch(value < 1, logcdf, 0), value >= 0, a > 0, b > 0)


class Exponential(PositiveContinuous):
    r"""
    Exponential log-likelihood.

    The pdf of this distribution is

    .. math::

       f(x \mid \lambda) = \lambda \exp\left\{ -\lambda x \right\}

    .. plot::

        import matplotlib.pyplot as plt
        import numpy as np
        import scipy.stats as st
        import arviz as az
        plt.style.use('arviz-darkgrid')
        x = np.linspace(0, 3, 100)
        for lam in [0.5, 1., 2.]:
            pdf = st.expon.pdf(x, scale=1.0/lam)
            plt.plot(x, pdf, label=r'$\lambda$ = {}'.format(lam))
        plt.xlabel('x', fontsize=12)
        plt.ylabel('f(x)', fontsize=12)
        plt.legend(loc=1)
        plt.show()

    ========  ============================
    Support   :math:`x \in [0, \infty)`
    Mean      :math:`\dfrac{1}{\lambda}`
    Variance  :math:`\dfrac{1}{\lambda^2}`
    ========  ============================

    Parameters
    ----------
    lam: float
        Rate or inverse scale (lam > 0)
    """
    rv_op = exponential

    @classmethod
    def dist(cls, lam, *args, **kwargs):
        lam = at.as_tensor_variable(floatX(lam))

        assert_negative_support(lam, "lam", "Exponential")

        # Aesara exponential op is parametrized in terms of mu (1/lam)
        return super().dist([at.inv(lam)], **kwargs)

    def logp(value, mu):
        """
        Calculate log-probability of Exponential distribution at specified value.

        Parameters
        ----------
        value: numeric
            Value(s) for which log-probability is calculated. If the log
            probabilities for multiple values are desired the values must be
            provided in a numpy array or Aesara tensor

        Returns
        -------
        TensorVariable
        """
        lam = at.inv(mu)
        return bound(
            at.log(lam) - lam * value,
            value >= 0,
            lam > 0,
        )

    def logcdf(value, mu):
        r"""
        Compute the log of cumulative distribution function for the Exponential distribution
        at the specified value.

        Parameters
        ----------
        value: numeric or np.ndarray or aesara.tensor
            Value(s) for which log CDF is calculated. If the log CDF for
            multiple values are desired the values must be provided in a numpy
            array or Aesara tensor.

        Returns
        -------
        TensorVariable
        """
        lam = at.inv(mu)
        return bound(
            at.log1mexp(-lam * value),
            0 <= value,
            0 <= lam,
        )


class Laplace(Continuous):
    r"""
    Laplace log-likelihood.

    The pdf of this distribution is

    .. math::

       f(x \mid \mu, b) =
           \frac{1}{2b} \exp \left\{ - \frac{|x - \mu|}{b} \right\}

    .. plot::

        import matplotlib.pyplot as plt
        import numpy as np
        import scipy.stats as st
        import arviz as az
        plt.style.use('arviz-darkgrid')
        x = np.linspace(-10, 10, 1000)
        mus = [0., 0., 0., -5.]
        bs = [1., 2., 4., 4.]
        for mu, b in zip(mus, bs):
            pdf = st.laplace.pdf(x, loc=mu, scale=b)
            plt.plot(x, pdf, label=r'$\mu$ = {}, $b$ = {}'.format(mu, b))
        plt.xlabel('x', fontsize=12)
        plt.ylabel('f(x)', fontsize=12)
        plt.legend(loc=1)
        plt.show()

    ========  ========================
    Support   :math:`x \in \mathbb{R}`
    Mean      :math:`\mu`
    Variance  :math:`2 b^2`
    ========  ========================

    Parameters
    ----------
    mu: float
        Location parameter.
    b: float
        Scale parameter (b > 0).
    """
    rv_op = laplace

    @classmethod
    def dist(cls, mu, b, *args, **kwargs):
        b = at.as_tensor_variable(floatX(b))
        mu = at.as_tensor_variable(floatX(mu))

        assert_negative_support(b, "b", "Laplace")
        return super().dist([mu, b], *args, **kwargs)

    def logp(value, mu, b):
        """
        Calculate log-probability of Laplace distribution at specified value.

        Parameters
        ----------
        value: numeric
            Value(s) for which log-probability is calculated. If the log probabilities for multiple
            values are desired the values must be provided in a numpy array or Aesara tensor

        Returns
        -------
        TensorVariable
        """
        return -at.log(2 * b) - abs(value - mu) / b

    def logcdf(value, mu, b):
        """
        Compute the log of the cumulative distribution function for Laplace distribution
        at the specified value.

        Parameters
        ----------
        value: numeric or np.ndarray or aesara.tensor
            Value(s) for which log CDF is calculated. If the log CDF for multiple
            values are desired the values must be provided in a numpy array or Aesara tensor.

        Returns
        -------
        TensorVariable
        """
        y = (value - mu) / b
        return bound(
            at.switch(
                at.le(value, mu),
                at.log(0.5) + y,
                at.switch(
                    at.gt(y, 1),
                    at.log1p(-0.5 * at.exp(-y)),
                    at.log(1 - 0.5 * at.exp(-y)),
                ),
            ),
            0 < b,
        )


class AsymmetricLaplaceRV(RandomVariable):
    name = "asymmetriclaplace"
    ndim_supp = 0
    ndims_params = [0, 0, 0]
    dtype = "floatX"
    _print_name = ("AsymmetricLaplace", "\\operatorname{AsymmetricLaplace}")

    @classmethod
    def rng_fn(cls, rng, b, kappa, mu, size=None):
        u = rng.uniform(size=size)
        switch = kappa ** 2 / (1 + kappa ** 2)
        non_positive_x = mu + kappa * np.log(u * (1 / switch)) / b
        positive_x = mu - np.log((1 - u) * (1 + kappa ** 2)) / (kappa * b)
        draws = non_positive_x * (u <= switch) + positive_x * (u > switch)
        return draws


asymmetriclaplace = AsymmetricLaplaceRV()


class AsymmetricLaplace(Continuous):
    r"""
    Asymmetric-Laplace log-likelihood.

    The pdf of this distribution is

    .. math::
        {f(x|\\b,\kappa,\mu) =
            \left({\frac{\\b}{\kappa + 1/\kappa}}\right)\,e^{-(x-\mu)\\b\,s\kappa ^{s}}}

    where

    .. math::

        s = sgn(x-\mu)

    ========  ========================
    Support   :math:`x \in \mathbb{R}`
    Mean      :math:`\mu-\frac{\\\kappa-1/\kappa}b`
    Variance  :math:`\frac{1+\kappa^{4}}{b^2\kappa^2 }`
    ========  ========================

    Parameters
    ----------
    b: float
        Scale parameter (b > 0)
    kappa: float
        Symmetry parameter (kappa > 0)
    mu: float
        Location parameter

    See Also:
    --------
    `Reference <https://en.wikipedia.org/wiki/Asymmetric_Laplace_distribution>`_
    """
    rv_op = asymmetriclaplace

    @classmethod
    def dist(cls, b, kappa, mu=0, *args, **kwargs):
        b = at.as_tensor_variable(floatX(b))
        kappa = at.as_tensor_variable(floatX(kappa))
        mu = mu = at.as_tensor_variable(floatX(mu))

        # mean = mu - (kappa - 1 / kappa) / b
        # variance = (1 + kappa ** 4) / (kappa ** 2 * b ** 2)

        assert_negative_support(kappa, "kappa", "AsymmetricLaplace")
        assert_negative_support(b, "b", "AsymmetricLaplace")

        return super().dist([b, kappa, mu], *args, **kwargs)

    def logp(value, b, kappa, mu):
        """
        Calculate log-probability of Asymmetric-Laplace distribution at specified value.

        Parameters
        ----------
        value: numeric
            Value(s) for which log-probability is calculated. If the log probabilities for multiple
            values are desired the values must be provided in a numpy array or Aesara tensor

        Returns
        -------
        TensorVariable
        """
        value = value - mu
        return bound(
            at.log(b / (kappa + (kappa ** -1)))
            + (-value * b * at.sgn(value) * (kappa ** at.sgn(value))),
            0 < b,
            0 < kappa,
        )


class Lognormal(PositiveContinuous):
    r"""
    Log-normal log-likelihood.

    Distribution of any random variable whose logarithm is normally
    distributed. A variable might be modeled as log-normal if it can
    be thought of as the multiplicative product of many small
    independent factors.

    The pdf of this distribution is

    .. math::

       f(x \mid \mu, \tau) =
           \frac{1}{x} \sqrt{\frac{\tau}{2\pi}}
           \exp\left\{ -\frac{\tau}{2} (\ln(x)-\mu)^2 \right\}

    .. plot::

        import matplotlib.pyplot as plt
        import numpy as np
        import scipy.stats as st
        import arviz as az
        plt.style.use('arviz-darkgrid')
        x = np.linspace(0, 3, 100)
        mus = [0., 0., 0.]
        sigmas = [.25, .5, 1.]
        for mu, sigma in zip(mus, sigmas):
            pdf = st.lognorm.pdf(x, sigma, scale=np.exp(mu))
            plt.plot(x, pdf, label=r'$\mu$ = {}, $\sigma$ = {}'.format(mu, sigma))
        plt.xlabel('x', fontsize=12)
        plt.ylabel('f(x)', fontsize=12)
        plt.legend(loc=1)
        plt.show()

    ========  =========================================================================
    Support   :math:`x \in [0, \infty)`
    Mean      :math:`\exp\{\mu + \frac{1}{2\tau}\}`
    Variance  :math:`(\exp\{\frac{1}{\tau}\} - 1) \times \exp\{2\mu + \frac{1}{\tau}\}`
    ========  =========================================================================

    Parameters
    ----------
    mu: float
        Location parameter.
    sigma: float
        Standard deviation. (sigma > 0). (only required if tau is not specified).
    tau: float
        Scale parameter (tau > 0). (only required if sigma is not specified).

    Examples
    --------

    .. code-block:: python

        # Example to show that we pass in only ``sigma`` or ``tau`` but not both.
        with pm.Model():
            x = pm.Lognormal('x', mu=2, sigma=30)

        with pm.Model():
            x = pm.Lognormal('x', mu=2, tau=1/100)
    """

    rv_op = lognormal

    @classmethod
    def dist(cls, mu=0, sigma=None, tau=None, sd=None, *args, **kwargs):
        if sd is not None:
            sigma = sd

        tau, sigma = get_tau_sigma(tau=tau, sigma=sigma)

        mu = at.as_tensor_variable(floatX(mu))
        sigma = at.as_tensor_variable(floatX(sigma))

        assert_negative_support(tau, "tau", "LogNormal")
        assert_negative_support(sigma, "sigma", "LogNormal")

        return super().dist([mu, sigma], *args, **kwargs)

    def logp(value, mu, sigma):
        """
        Calculate log-probability of Lognormal distribution at specified value.

        Parameters
        ----------
        value: numeric
            Value(s) for which log-probability is calculated. If the log probabilities for multiple
            values are desired the values must be provided in a numpy array or Aesara tensor

        Returns
        -------
        TensorVariable
        """
        tau, sigma = get_tau_sigma(tau=None, sigma=sigma)
        return bound(
            -0.5 * tau * (at.log(value) - mu) ** 2
            + 0.5 * at.log(tau / (2.0 * np.pi))
            - at.log(value),
            tau > 0,
        )

    def logcdf(value, mu, sigma):
        """
        Compute the log of the cumulative distribution function for Lognormal distribution
        at the specified value.

        Parameters
        ----------
        value: numeric or np.ndarray or aesara.tensor
            Value(s) for which log CDF is calculated. If the log CDF for multiple
            values are desired the values must be provided in a numpy array or Aesara tensor.

        Returns
        -------
        TensorVariable
        """

        return bound(
            normal_lcdf(mu, sigma, at.log(value)),
            0 < value,
            0 < sigma,
        )


class StudentTRV(RandomVariable):
    name = "studentt"
    ndim_supp = 0
    ndims_params = [0, 0, 0]
    dtype = "floatX"
    _print_name = ("StudentT", "\\operatorname{StudentT}")

    @classmethod
    def rng_fn(cls, rng, nu, mu, sigma, size=None):
        return stats.t.rvs(nu, mu, sigma, size=size, random_state=rng)


studentt = StudentTRV()


class StudentT(Continuous):
    r"""
    Student's T log-likelihood.

    Describes a normal variable whose precision is gamma distributed.
    If only nu parameter is passed, this specifies a standard (central)
    Student's T.

    The pdf of this distribution is

    .. math::

       f(x|\mu,\lambda,\nu) =
           \frac{\Gamma(\frac{\nu + 1}{2})}{\Gamma(\frac{\nu}{2})}
           \left(\frac{\lambda}{\pi\nu}\right)^{\frac{1}{2}}
           \left[1+\frac{\lambda(x-\mu)^2}{\nu}\right]^{-\frac{\nu+1}{2}}

    .. plot::

        import matplotlib.pyplot as plt
        import numpy as np
        import scipy.stats as st
        import arviz as az
        plt.style.use('arviz-darkgrid')
        x = np.linspace(-8, 8, 200)
        mus = [0., 0., -2., -2.]
        sigmas = [1., 1., 1., 2.]
        dfs = [1., 5., 5., 5.]
        for mu, sigma, df in zip(mus, sigmas, dfs):
            pdf = st.t.pdf(x, df, loc=mu, scale=sigma)
            plt.plot(x, pdf, label=r'$\mu$ = {}, $\sigma$ = {}, $\nu$ = {}'.format(mu, sigma, df))
        plt.xlabel('x', fontsize=12)
        plt.ylabel('f(x)', fontsize=12)
        plt.legend(loc=1)
        plt.show()

    ========  ========================
    Support   :math:`x \in \mathbb{R}`
    ========  ========================

    Parameters
    ----------
    nu: float
        Degrees of freedom, also known as normality parameter (nu > 0).
    mu: float
        Location parameter.
    sigma: float
        Scale parameter (sigma > 0). Converges to the standard deviation as nu
        increases. (only required if lam is not specified)
    lam: float
        Scale parameter (lam > 0). Converges to the precision as nu
        increases. (only required if sigma is not specified)

    Examples
    --------
    .. code-block:: python

        with pm.Model():
            x = pm.StudentT('x', nu=15, mu=0, sigma=10)

        with pm.Model():
            x = pm.StudentT('x', nu=15, mu=0, lam=1/23)
    """
    rv_op = studentt

    @classmethod
    def dist(cls, nu, mu=0, lam=None, sigma=None, sd=None, *args, **kwargs):
        if sd is not None:
            sigma = sd
        nu = at.as_tensor_variable(floatX(nu))
        lam, sigma = get_tau_sigma(tau=lam, sigma=sigma)
        sigma = at.as_tensor_variable(sigma)

        assert_negative_support(sigma, "sigma (lam)", "StudentT")
        assert_negative_support(nu, "nu", "StudentT")

        return super().dist([nu, mu, sigma], **kwargs)

    def logp(value, nu, mu, sigma):
        """
        Calculate log-probability of StudentT distribution at specified value.

        Parameters
        ----------
        value: numeric
            Value(s) for which log-probability is calculated. If the log probabilities for multiple
            values are desired the values must be provided in a numpy array or Aesara tensor

        Returns
        -------
        TensorVariable
        """
        lam, sigma = get_tau_sigma(sigma=sigma)
        return bound(
            gammaln((nu + 1.0) / 2.0)
            + 0.5 * at.log(lam / (nu * np.pi))
            - gammaln(nu / 2.0)
            - (nu + 1.0) / 2.0 * at.log1p(lam * (value - mu) ** 2 / nu),
            lam > 0,
            nu > 0,
            sigma > 0,
        )

    def logcdf(value, nu, mu, sigma):
        """
        Compute the log of the cumulative distribution function for Student's T distribution
        at the specified value.

        Parameters
        ----------
        value: numeric or np.ndarray or aesara.tensor
            Value(s) for which log CDF is calculated. If the log CDF for multiple
            values are desired the values must be provided in a numpy array or Aesara tensor.

        Returns
        -------
        TensorVariable
        """
        lam, sigma = get_tau_sigma(sigma=sigma)

        t = (value - mu) / sigma
        sqrt_t2_nu = at.sqrt(t ** 2 + nu)
        x = (t + sqrt_t2_nu) / (2.0 * sqrt_t2_nu)

        return bound(
            at.log(at.betainc(nu / 2.0, nu / 2.0, x)),
            0 < nu,
            0 < sigma,
            0 < lam,
        )


class Pareto(BoundedContinuous):
    r"""
    Pareto log-likelihood.

    Often used to characterize wealth distribution, or other examples of the
    80/20 rule.

    The pdf of this distribution is

    .. math::

       f(x \mid \alpha, m) = \frac{\alpha m^{\alpha}}{x^{\alpha+1}}

    .. plot::

        import matplotlib.pyplot as plt
        import numpy as np
        import scipy.stats as st
        import arviz as az
        plt.style.use('arviz-darkgrid')
        x = np.linspace(0, 4, 1000)
        alphas = [1., 2., 5., 5.]
        ms = [1., 1., 1., 2.]
        for alpha, m in zip(alphas, ms):
            pdf = st.pareto.pdf(x, alpha, scale=m)
            plt.plot(x, pdf, label=r'$\alpha$ = {}, m = {}'.format(alpha, m))
        plt.xlabel('x', fontsize=12)
        plt.ylabel('f(x)', fontsize=12)
        plt.legend(loc=1)
        plt.show()

    ========  =============================================================
    Support   :math:`x \in [m, \infty)`
    Mean      :math:`\dfrac{\alpha m}{\alpha - 1}` for :math:`\alpha \ge 1`
    Variance  :math:`\dfrac{m \alpha}{(\alpha - 1)^2 (\alpha - 2)}`
              for :math:`\alpha > 2`
    ========  =============================================================

    Parameters
    ----------
    alpha: float
        Shape parameter (alpha > 0).
    m: float
        Scale parameter (m > 0).
    """
    rv_op = pareto
    bound_args_indices = (1, None)  # lower-bounded by `m`

    @classmethod
    def dist(
        cls, alpha: float = None, m: float = None, no_assert: bool = False, **kwargs
    ) -> RandomVariable:
        alpha = at.as_tensor_variable(floatX(alpha))
        m = at.as_tensor_variable(floatX(m))

        assert_negative_support(alpha, "alpha", "Pareto")
        assert_negative_support(m, "m", "Pareto")

        return super().dist([alpha, m], **kwargs)

    def logp(
        value: Union[float, np.ndarray, TensorVariable],
        alpha: Union[float, np.ndarray, TensorVariable],
        m: Union[float, np.ndarray, TensorVariable],
    ):
        """
        Calculate log-probability of Pareto distribution at specified value.

        Parameters
        ----------
        value: numeric
            Value(s) for which log-probability is calculated. If the log probabilities for multiple
            values are desired the values must be provided in a numpy array or Aesara tensor

        Returns
        -------
        TensorVariable
        """
        return bound(
            at.log(alpha) + logpow(m, alpha) - logpow(value, alpha + 1),
            value >= m,
            alpha > 0,
            m > 0,
        )

    def _distr_parameters_for_repr(self):
        return ["alpha", "m"]

    def logcdf(
        value: Union[float, np.ndarray, TensorVariable],
        alpha: Union[float, np.ndarray, TensorVariable],
        m: Union[float, np.ndarray, TensorVariable],
    ):
        """
        Compute the log of the cumulative distribution function for Pareto distribution
        at the specified value.

        Parameters
        ----------
        value: numeric or np.ndarray or aesara.tensor
            Value(s) for which log CDF is calculated. If the log CDF for multiple
            values are desired the values must be provided in a numpy array or Aesara tensor.

        Returns
        -------
        TensorVariable
        """
        arg = (m / value) ** alpha
        return bound(
            at.switch(
                at.le(arg, 1e-5),
                at.log1p(-arg),
                at.log(1 - arg),
            ),
            m <= value,
            0 < alpha,
            0 < m,
        )


class Cauchy(Continuous):
    r"""
    Cauchy log-likelihood.

    Also known as the Lorentz or the Breit-Wigner distribution.

    The pdf of this distribution is

    .. math::

       f(x \mid \alpha, \beta) =
           \frac{1}{\pi \beta [1 + (\frac{x-\alpha}{\beta})^2]}

    .. plot::

        import matplotlib.pyplot as plt
        import numpy as np
        import scipy.stats as st
        import arviz as az
        plt.style.use('arviz-darkgrid')
        x = np.linspace(-5, 5, 500)
        alphas = [0., 0., 0., -2.]
        betas = [.5, 1., 2., 1.]
        for a, b in zip(alphas, betas):
            pdf = st.cauchy.pdf(x, loc=a, scale=b)
            plt.plot(x, pdf, label=r'$\alpha$ = {}, $\beta$ = {}'.format(a, b))
        plt.xlabel('x', fontsize=12)
        plt.ylabel('f(x)', fontsize=12)
        plt.legend(loc=1)
        plt.show()

    ========  ========================
    Support   :math:`x \in \mathbb{R}`
    Mode      :math:`\alpha`
    Mean      undefined
    Variance  undefined
    ========  ========================

    Parameters
    ----------
    alpha: float
        Location parameter
    beta: float
        Scale parameter > 0
    """
    rv_op = cauchy

    @classmethod
    def dist(cls, alpha, beta, *args, **kwargs):
        alpha = at.as_tensor_variable(floatX(alpha))
        beta = at.as_tensor_variable(floatX(beta))

        # median = alpha
        # mode = alpha

        assert_negative_support(beta, "beta", "Cauchy")
        return super().dist([alpha, beta], **kwargs)

    def logp(value, alpha, beta):
        """
        Calculate log-probability of Cauchy distribution at specified value.

        Parameters
        ----------
        value: numeric
            Value(s) for which log-probability is calculated. If the log probabilities for multiple
            values are desired the values must be provided in a numpy array or Aesara tensor

        Returns
        -------
        TensorVariable
        """
        return bound(
            -at.log(np.pi) - at.log(beta) - at.log1p(((value - alpha) / beta) ** 2), beta > 0
        )

    def logcdf(value, alpha, beta):
        """
        Compute the log of the cumulative distribution function for Cauchy distribution
        at the specified value.

        Parameters
        ----------
        value: numeric or np.ndarray or aesara.tensor
            Value(s) for which log CDF is calculated. If the log CDF for multiple
            values are desired the values must be provided in a numpy array or Aesara tensor.

        Returns
        -------
        TensorVariable
        """
        return bound(
            at.log(0.5 + at.arctan((value - alpha) / beta) / np.pi),
            0 < beta,
        )


class HalfCauchy(PositiveContinuous):
    r"""
    Half-Cauchy log-likelihood.

    The pdf of this distribution is

    .. math::

       f(x \mid \beta) = \frac{2}{\pi \beta [1 + (\frac{x}{\beta})^2]}

    .. plot::

        import matplotlib.pyplot as plt
        import numpy as np
        import scipy.stats as st
        import arviz as az
        plt.style.use('arviz-darkgrid')
        x = np.linspace(0, 5, 200)
        for b in [0.5, 1.0, 2.0]:
            pdf = st.cauchy.pdf(x, scale=b)
            plt.plot(x, pdf, label=r'$\beta$ = {}'.format(b))
        plt.xlabel('x', fontsize=12)
        plt.ylabel('f(x)', fontsize=12)
        plt.legend(loc=1)
        plt.show()

    ========  ========================
    Support   :math:`x \in [0, \infty)`
    Mode      0
    Mean      undefined
    Variance  undefined
    ========  ========================

    Parameters
    ----------
    beta: float
        Scale parameter (beta > 0).
    """
    rv_op = halfcauchy

    @classmethod
    def dist(cls, beta, *args, **kwargs):
        beta = at.as_tensor_variable(floatX(beta))
        assert_negative_support(beta, "beta", "HalfCauchy")
        return super().dist([0.0, beta], **kwargs)

    def logp(value, loc, beta):
        """
        Calculate log-probability of HalfCauchy distribution at specified value.

        Parameters
        ----------
        value: numeric
            Value(s) for which log-probability is calculated. If the log probabilities for multiple
            values are desired the values must be provided in a numpy array or Aesara tensor

        Returns
        -------
        TensorVariable
        """
        return bound(
            at.log(2) - at.log(np.pi) - at.log(beta) - at.log1p(((value - loc) / beta) ** 2),
            value >= loc,
            beta > 0,
        )

    def logcdf(value, loc, beta):
        """
        Compute the log of the cumulative distribution function for HalfCauchy distribution
        at the specified value.

        Parameters
        ----------
        value: numeric or np.ndarray or aesara.tensor
            Value(s) for which log CDF is calculated. If the log CDF for multiple
            values are desired the values must be provided in a numpy array or Aesara tensor.

        Returns
        -------
        TensorVariable
        """
        return bound(
            at.log(2 * at.arctan((value - loc) / beta) / np.pi),
            loc <= value,
            0 < beta,
        )


class Gamma(PositiveContinuous):
    r"""
    Gamma log-likelihood.

    Represents the sum of alpha exponentially distributed random variables,
    each of which has rate beta.

    The pdf of this distribution is

    .. math::

       f(x \mid \alpha, \beta) =
           \frac{\beta^{\alpha}x^{\alpha-1}e^{-\beta x}}{\Gamma(\alpha)}

    .. plot::

        import matplotlib.pyplot as plt
        import numpy as np
        import scipy.stats as st
        import arviz as az
        plt.style.use('arviz-darkgrid')
        x = np.linspace(0, 20, 200)
        alphas = [1., 2., 3., 7.5]
        betas = [.5, .5, 1., 1.]
        for a, b in zip(alphas, betas):
            pdf = st.gamma.pdf(x, a, scale=1.0/b)
            plt.plot(x, pdf, label=r'$\alpha$ = {}, $\beta$ = {}'.format(a, b))
        plt.xlabel('x', fontsize=12)
        plt.ylabel('f(x)', fontsize=12)
        plt.legend(loc=1)
        plt.show()

    ========  ===============================
    Support   :math:`x \in (0, \infty)`
    Mean      :math:`\dfrac{\alpha}{\beta}`
    Variance  :math:`\dfrac{\alpha}{\beta^2}`
    ========  ===============================

    Gamma distribution can be parameterized either in terms of alpha and
    beta or mean and standard deviation. The link between the two
    parametrizations is given by

    .. math::

       \alpha &= \frac{\mu^2}{\sigma^2} \\
       \beta &= \frac{\mu}{\sigma^2}

    Parameters
    ----------
    alpha: float
        Shape parameter (alpha > 0).
    beta: float
        Rate parameter (beta > 0).
    mu: float
        Alternative shape parameter (mu > 0).
    sigma: float
        Alternative scale parameter (sigma > 0).
    """
    rv_op = gamma

    @classmethod
    def dist(cls, alpha=None, beta=None, mu=None, sigma=None, sd=None, no_assert=False, **kwargs):
        if sd is not None:
            sigma = sd

        alpha, beta = cls.get_alpha_beta(alpha, beta, mu, sigma)
        alpha = at.as_tensor_variable(floatX(alpha))
        beta = at.as_tensor_variable(floatX(beta))

        if not no_assert:
            assert_negative_support(alpha, "alpha", "Gamma")
            assert_negative_support(beta, "beta", "Gamma")

        # The Aesara `GammaRV` `Op` will invert the `beta` parameter itself
        return super().dist([alpha, beta], **kwargs)

    @classmethod
    def get_alpha_beta(cls, alpha=None, beta=None, mu=None, sigma=None):
        if (alpha is not None) and (beta is not None):
            pass
        elif (mu is not None) and (sigma is not None):
            alpha = mu ** 2 / sigma ** 2
            beta = mu / sigma ** 2
        else:
            raise ValueError(
                "Incompatible parameterization. Either use "
                "alpha and beta, or mu and sigma to specify "
                "distribution."
            )

        return alpha, beta

    def logp(value, alpha, inv_beta):
        """
        Calculate log-probability of Gamma distribution at specified value.

        Parameters
        ----------
        value: numeric
            Value(s) for which log-probability is calculated. If the log
            probabilities for multiple values are desired the values must be
            provided in a numpy array or `TensorVariable`.

        Returns
        -------
        TensorVariable
        """
        beta = at.inv(inv_beta)
        return bound(
            -gammaln(alpha) + logpow(beta, alpha) - beta * value + logpow(value, alpha - 1),
            value >= 0,
            alpha > 0,
            beta > 0,
        )

    def logcdf(value, alpha, inv_beta):
        """
        Compute the log of the cumulative distribution function for Gamma distribution
        at the specified value.

        Parameters
        ----------
        value: numeric or np.ndarray or `TensorVariable`
            Value(s) for which log CDF is calculated. If the log CDF for
            multiple values are desired the values must be provided in a numpy
            array or `TensorVariable`.

        Returns
        -------
        TensorVariable
        """
        beta = at.inv(inv_beta)

        return bound(
            at.log(at.gammainc(alpha, beta * value)),
            0 <= value,
            0 < alpha,
            0 < beta,
        )


class InverseGamma(PositiveContinuous):
    r"""
    Inverse gamma log-likelihood, the reciprocal of the gamma distribution.

    The pdf of this distribution is

    .. math::

       f(x \mid \alpha, \beta) =
           \frac{\beta^{\alpha}}{\Gamma(\alpha)} x^{-\alpha - 1}
           \exp\left(\frac{-\beta}{x}\right)

    .. plot::

        import matplotlib.pyplot as plt
        import numpy as np
        import scipy.stats as st
        import arviz as az
        plt.style.use('arviz-darkgrid')
        x = np.linspace(0, 3, 500)
        alphas = [1., 2., 3., 3.]
        betas = [1., 1., 1., .5]
        for a, b in zip(alphas, betas):
            pdf = st.invgamma.pdf(x, a, scale=b)
            plt.plot(x, pdf, label=r'$\alpha$ = {}, $\beta$ = {}'.format(a, b))
        plt.xlabel('x', fontsize=12)
        plt.ylabel('f(x)', fontsize=12)
        plt.legend(loc=1)
        plt.show()

    ========  ======================================================
    Support   :math:`x \in (0, \infty)`
    Mean      :math:`\dfrac{\beta}{\alpha-1}` for :math:`\alpha > 1`
    Variance  :math:`\dfrac{\beta^2}{(\alpha-1)^2(\alpha - 2)}`
              for :math:`\alpha > 2`
    ========  ======================================================

    Parameters
    ----------
    alpha: float
        Shape parameter (alpha > 0).
    beta: float
        Scale parameter (beta > 0).
    mu: float
        Alternative shape parameter (mu > 0).
    sigma: float
        Alternative scale parameter (sigma > 0).
    """
    rv_op = invgamma

    @classmethod
    def dist(cls, alpha=None, beta=None, mu=None, sigma=None, sd=None, *args, **kwargs):
        if sd is not None:
            sigma = sd

        alpha, beta = cls._get_alpha_beta(alpha, beta, mu, sigma)
        alpha = at.as_tensor_variable(floatX(alpha))
        beta = at.as_tensor_variable(floatX(beta))

        # m = beta / (alpha - 1.0)
        # try:
        #     mean = (alpha > 1) * m or np.inf
        # except ValueError:  # alpha is an array
        #     m[alpha <= 1] = np.inf
        #     mean = m

        # mode = beta / (alpha + 1.0)
        # variance = at.switch(
        #     at.gt(alpha, 2), (beta ** 2) / ((alpha - 2) * (alpha - 1.0) ** 2), np.inf
        # )

        assert_negative_support(alpha, "alpha", "InverseGamma")
        assert_negative_support(beta, "beta", "InverseGamma")

        return super().dist([alpha, beta], **kwargs)

    @classmethod
    def _get_alpha_beta(cls, alpha, beta, mu, sigma):
        if alpha is not None:
            if beta is not None:
                pass
            else:
                beta = 1
        elif (mu is not None) and (sigma is not None):
            alpha = (2 * sigma ** 2 + mu ** 2) / sigma ** 2
            beta = mu * (mu ** 2 + sigma ** 2) / sigma ** 2
        else:
            raise ValueError(
                "Incompatible parameterization. Either use "
                "alpha and (optionally) beta, or mu and sigma to specify "
                "distribution."
            )

        return alpha, beta

    @classmethod
    def _distr_parameters_for_repr(self):
        return ["alpha", "beta"]

    def logp(value, alpha, beta):
        """
        Calculate log-probability of InverseGamma distribution at specified value.

        Parameters
        ----------
        value: numeric
            Value(s) for which log-probability is calculated. If the log probabilities for multiple
            values are desired the values must be provided in a numpy array or Aesara tensor

        Returns
        -------
        TensorVariable
        """
        return bound(
            logpow(beta, alpha) - gammaln(alpha) - beta / value + logpow(value, -alpha - 1),
            value > 0,
            alpha > 0,
            beta > 0,
        )

    def logcdf(value, alpha, beta):
        """
        Compute the log of the cumulative distribution function for Inverse Gamma distribution
        at the specified value.

        Parameters
        ----------
        value: numeric or np.ndarray or aesara.tensor
            Value(s) for which log CDF is calculated. If the log CDF for multiple
            values are desired the values must be provided in a numpy array or Aesara tensor.

        Returns
        -------
        TensorVariable
        """

        return bound(
            at.log(at.gammaincc(alpha, beta / value)),
            0 <= value,
            0 < alpha,
            0 < beta,
        )


class ChiSquared(PositiveContinuous):
    r"""
    :math:`\chi^2` log-likelihood.

    The pdf of this distribution is

    .. math::

       f(x \mid \nu) = \frac{x^{(\nu-2)/2}e^{-x/2}}{2^{\nu/2}\Gamma(\nu/2)}

    .. plot::

        import matplotlib.pyplot as plt
        import numpy as np
        import scipy.stats as st
        import arviz as az
        plt.style.use('arviz-darkgrid')
        x = np.linspace(0, 15, 200)
        for df in [1, 2, 3, 6, 9]:
            pdf = st.chi2.pdf(x, df)
            plt.plot(x, pdf, label=r'$\nu$ = {}'.format(df))
        plt.xlabel('x', fontsize=12)
        plt.ylabel('f(x)', fontsize=12)
        plt.ylim(0, 0.6)
        plt.legend(loc=1)
        plt.show()

    ========  ===============================
    Support   :math:`x \in [0, \infty)`
    Mean      :math:`\nu`
    Variance  :math:`2 \nu`
    ========  ===============================

    Parameters
    ----------
    nu: float
        Degrees of freedom (nu > 0).
    """
    rv_op = chisquare

    @classmethod
    def dist(cls, nu, *args, **kwargs):
        nu = at.as_tensor_variable(floatX(nu))
        return super().dist([nu], *args, **kwargs)

    def logp(value, nu):
        """
        Calculate log-probability of ChiSquared distribution at specified value.

        Parameters
        ----------
        value: numeric
            Value(s) for which log-probability is calculated. If the log probabilities for multiple
            values are desired the values must be provided in a numpy array or Aesara tensor

        Returns
        -------
        TensorVariable
        """
        return Gamma.logp(value, nu / 2, 2)

    def logcdf(value, nu):
        """
        Compute the log of the cumulative distribution function for ChiSquared distribution
        at the specified value.

        Parameters
        ----------
        value: numeric or np.ndarray or `TensorVariable`
            Value(s) for which log CDF is calculated. If the log CDF for
            multiple values are desired the values must be provided in a numpy
            array or `TensorVariable`.
        Returns
        -------
        TensorVariable
        """
        return Gamma.logcdf(value, nu / 2, 2)


# TODO: Remove this once logpt for multiplication is working!
class WeibullBetaRV(WeibullRV):
    ndims_params = [0, 0]

    @classmethod
    def rng_fn(cls, rng, alpha, beta, size):
        return beta * rng.weibull(alpha, size=size)


weibull_beta = WeibullBetaRV()


class Weibull(PositiveContinuous):
    r"""
    Weibull log-likelihood.

    The pdf of this distribution is

    .. math::

       f(x \mid \alpha, \beta) =
           \frac{\alpha x^{\alpha - 1}
           \exp(-(\frac{x}{\beta})^{\alpha})}{\beta^\alpha}

    .. plot::

        import matplotlib.pyplot as plt
        import numpy as np
        import scipy.stats as st
        import arviz as az
        plt.style.use('arviz-darkgrid')
        x = np.linspace(0, 3, 200)
        alphas = [.5, 1., 1.5, 5., 5.]
        betas = [1., 1., 1., 1.,  2]
        for a, b in zip(alphas, betas):
            pdf = st.weibull_min.pdf(x, a, scale=b)
            plt.plot(x, pdf, label=r'$\alpha$ = {}, $\beta$ = {}'.format(a, b))
        plt.xlabel('x', fontsize=12)
        plt.ylabel('f(x)', fontsize=12)
        plt.ylim(0, 2.5)
        plt.legend(loc=1)
        plt.show()

    ========  ====================================================
    Support   :math:`x \in [0, \infty)`
    Mean      :math:`\beta \Gamma(1 + \frac{1}{\alpha})`
    Variance  :math:`\beta^2 \Gamma(1 + \frac{2}{\alpha} - \mu^2/\beta^2)`
    ========  ====================================================

    Parameters
    ----------
    alpha: float
        Shape parameter (alpha > 0).
    beta: float
        Scale parameter (beta > 0).
    """

    rv_op = weibull_beta

    @classmethod
    def dist(cls, alpha, beta, *args, **kwargs):
        alpha = at.as_tensor_variable(floatX(alpha))
        beta = at.as_tensor_variable(floatX(beta))

        assert_negative_support(alpha, "alpha", "Weibull")
        assert_negative_support(beta, "beta", "Weibull")

        return super().dist([alpha, beta], *args, **kwargs)

    def logp(value, alpha, beta):
        """
        Calculate log-probability of Weibull distribution at specified value.

        Parameters
        ----------
        value: numeric
            Value(s) for which log-probability is calculated. If the log probabilities for multiple
            values are desired the values must be provided in a numpy array or Aesara tensor

        Returns
        -------
        TensorVariable
        """
        return bound(
            at.log(alpha)
            - at.log(beta)
            + (alpha - 1) * at.log(value / beta)
            - (value / beta) ** alpha,
            value >= 0,
            alpha > 0,
            beta > 0,
        )

    def logcdf(value, alpha, beta):
        r"""
        Compute the log of the cumulative distribution function for Weibull distribution
        at the specified value.

        Parameters
        ----------
        value: numeric or np.ndarray or aesara.tensor
            Value(s) for which log CDF is calculated. If the log CDF for multiple
            values are desired the values must be provided in a numpy array or Aesara tensor.

        Returns
        -------
        TensorVariable
        """
        a = (value / beta) ** alpha
        return bound(
            at.log1mexp(-a),
            0 <= value,
            0 < alpha,
            0 < beta,
        )


class HalfStudentTRV(RandomVariable):
    name = "halfstudentt"
    ndim_supp = 0
    ndims_params = [0, 0]
    dtype = "floatX"
    _print_name = ("HalfStudentT", "\\operatorname{HalfStudentT}")

    @classmethod
    def rng_fn(cls, rng, nu, sigma, size=None):
        return np.abs(stats.t.rvs(nu, sigma, size=size, random_state=rng))


halfstudentt = HalfStudentTRV()


class HalfStudentT(PositiveContinuous):
    r"""
    Half Student's T log-likelihood

    The pdf of this distribution is

    .. math::

        f(x \mid \sigma,\nu) =
            \frac{2\;\Gamma\left(\frac{\nu+1}{2}\right)}
            {\Gamma\left(\frac{\nu}{2}\right)\sqrt{\nu\pi\sigma^2}}
            \left(1+\frac{1}{\nu}\frac{x^2}{\sigma^2}\right)^{-\frac{\nu+1}{2}}

    .. plot::

        import matplotlib.pyplot as plt
        import numpy as np
        import scipy.stats as st
        import arviz as az
        plt.style.use('arviz-darkgrid')
        x = np.linspace(0, 5, 200)
        sigmas = [1., 1., 2., 1.]
        nus = [.5, 1., 1., 30.]
        for sigma, nu in zip(sigmas, nus):
            pdf = st.t.pdf(x, df=nu, loc=0, scale=sigma)
            plt.plot(x, pdf, label=r'$\sigma$ = {}, $\nu$ = {}'.format(sigma, nu))
        plt.xlabel('x', fontsize=12)
        plt.ylabel('f(x)', fontsize=12)
        plt.legend(loc=1)
        plt.show()

    ========  ========================
    Support   :math:`x \in [0, \infty)`
    ========  ========================

    Parameters
    ----------
    nu: float
        Degrees of freedom, also known as normality parameter (nu > 0).
    sigma: float
        Scale parameter (sigma > 0). Converges to the standard deviation as nu
        increases. (only required if lam is not specified)
    lam: float
        Scale parameter (lam > 0). Converges to the precision as nu
        increases. (only required if sigma is not specified)

    Examples
    --------
    .. code-block:: python

        # Only pass in one of lam or sigma, but not both.
        with pm.Model():
            x = pm.HalfStudentT('x', sigma=10, nu=10)

        with pm.Model():
            x = pm.HalfStudentT('x', lam=4, nu=10)
    """
    rv_op = halfstudentt

    @classmethod
    def dist(cls, nu=1, sigma=None, lam=None, sd=None, *args, **kwargs):

        if sd is not None:
            sigma = sd

        nu = at.as_tensor_variable(floatX(nu))
        lam, sigma = get_tau_sigma(lam, sigma)
        sigma = at.as_tensor_variable(sigma)

        # mode = at.as_tensor_variable(0)
        # median = at.as_tensor_variable(sigma)
        # sd = at.as_tensor_variable(sigma)

        assert_negative_support(nu, "nu", "HalfStudentT")
        assert_negative_support(lam, "lam", "HalfStudentT")
        assert_negative_support(sigma, "sigma", "HalfStudentT")

        return super().dist([nu, sigma], *args, **kwargs)

    def logp(value, nu, sigma):
        """
        Calculate log-probability of HalfStudentT distribution at specified value.

        Parameters
        ----------
        value: numeric
            Value(s) for which log-probability is calculated. If the log probabilities for multiple
            values are desired the values must be provided in a numpy array or Aesara tensor

        Returns
        -------
        TensorVariable
        """

        lam, sigma = get_tau_sigma(None, sigma)

        return bound(
            at.log(2)
            + gammaln((nu + 1.0) / 2.0)
            - gammaln(nu / 2.0)
            - 0.5 * at.log(nu * np.pi * sigma ** 2)
            - (nu + 1.0) / 2.0 * at.log1p(value ** 2 / (nu * sigma ** 2)),
            sigma > 0,
            lam > 0,
            nu > 0,
            value >= 0,
        )

    def _distr_parameters_for_repr(self):
        return ["nu", "lam"]


class ExGaussianRV(RandomVariable):
    name = "exgaussian"
    ndim_supp = 0
    ndims_params = [0, 0, 0]
    dtype = "floatX"
    _print_name = ("ExGaussian", "\\operatorname{ExGaussian}")

    @classmethod
    def rng_fn(cls, rng, mu, sigma, nu, size=None):
        return rng.normal(mu, sigma, size=size) + rng.exponential(scale=nu, size=size)


exgaussian = ExGaussianRV()


class ExGaussian(Continuous):
    r"""
    Exponentially modified Gaussian log-likelihood.

    Results from the convolution of a normal distribution with an exponential
    distribution.

    The pdf of this distribution is

    .. math::

       f(x \mid \mu, \sigma, \tau) =
           \frac{1}{\nu}\;
           \exp\left\{\frac{\mu-x}{\nu}+\frac{\sigma^2}{2\nu^2}\right\}
           \Phi\left(\frac{x-\mu}{\sigma}-\frac{\sigma}{\nu}\right)

    where :math:`\Phi` is the cumulative distribution function of the
    standard normal distribution.

    .. plot::

        import matplotlib.pyplot as plt
        import numpy as np
        import scipy.stats as st
        import arviz as az
        plt.style.use('arviz-darkgrid')
        x = np.linspace(-6, 9, 200)
        mus = [0., -2., 0., -3.]
        sigmas = [1., 1., 3., 1.]
        nus = [1., 1., 1., 4.]
        for mu, sigma, nu in zip(mus, sigmas, nus):
            pdf = st.exponnorm.pdf(x, nu/sigma, loc=mu, scale=sigma)
            plt.plot(x, pdf, label=r'$\mu$ = {}, $\sigma$ = {}, $\nu$ = {}'.format(mu, sigma, nu))
        plt.xlabel('x', fontsize=12)
        plt.ylabel('f(x)', fontsize=12)
        plt.legend(loc=1)
        plt.show()

    ========  ========================
    Support   :math:`x \in \mathbb{R}`
    Mean      :math:`\mu + \nu`
    Variance  :math:`\sigma^2 + \nu^2`
    ========  ========================

    Parameters
    ----------
    mu: float
        Mean of the normal distribution.
    sigma: float
        Standard deviation of the normal distribution (sigma > 0).
    nu: float
        Mean of the exponential distribution (nu > 0).

    References
    ----------
    .. [Rigby2005] Rigby R.A. and Stasinopoulos D.M. (2005).
        "Generalized additive models for location, scale and shape"
        Applied Statististics., 54, part 3, pp 507-554.

    .. [Lacouture2008] Lacouture, Y. and Couseanou, D. (2008).
        "How to use MATLAB to fit the ex-Gaussian and other probability
        functions to a distribution of response times".
        Tutorials in Quantitative Methods for Psychology,
        Vol. 4, No. 1, pp 35-45.
    """
    rv_op = exgaussian

    @classmethod
    def dist(cls, mu=0.0, sigma=None, nu=None, sd=None, *args, **kwargs):

        if sd is not None:
            sigma = sd

        mu = at.as_tensor_variable(floatX(mu))
        sigma = at.as_tensor_variable(floatX(sigma))
        nu = at.as_tensor_variable(floatX(nu))

        # sd = sigma
        # mean = mu + nu
        # variance = (sigma ** 2) + (nu ** 2)

        assert_negative_support(sigma, "sigma", "ExGaussian")
        assert_negative_support(nu, "nu", "ExGaussian")

        return super().dist([mu, sigma, nu], *args, **kwargs)

    def logp(value, mu, sigma, nu):
        """
        Calculate log-probability of ExGaussian distribution at specified value.

        Parameters
        ----------
        value: numeric
            Value(s) for which log-probability is calculated. If the log probabilities for multiple
            values are desired the values must be provided in a numpy array or Aesara tensor

        Returns
        -------
        TensorVariable
        """

        # Alogithm is adapted from dexGAUS.R from gamlss
        return bound(
            at.switch(
                at.gt(nu, 0.05 * sigma),
                (
                    -at.log(nu)
                    + (mu - value) / nu
                    + 0.5 * (sigma / nu) ** 2
                    + normal_lcdf(mu + (sigma ** 2) / nu, sigma, value)
                ),
                log_normal(value, mean=mu, sigma=sigma),
            ),
            0 < sigma,
            0 < nu,
        )

    def logcdf(value, mu, sigma, nu):
        """
        Compute the log of the cumulative distribution function for ExGaussian distribution
        at the specified value.

        References
        ----------
        .. [Rigby2005] R.A. Rigby (2005).
           "Generalized additive models for location, scale and shape"
           https://doi.org/10.1111/j.1467-9876.2005.00510.x

        Parameters
        ----------
        value: numeric or np.ndarray or aesara.tensor
            Value(s) for which log CDF is calculated. If the log CDF for multiple
            values are desired the values must be provided in a numpy array or Aesara tensor.

        Returns
        -------
        TensorVariable
        """

        # Alogithm is adapted from pexGAUS.R from gamlss
        return bound(
            at.switch(
                at.gt(nu, 0.05 * sigma),
                logdiffexp(
                    normal_lcdf(mu, sigma, value),
                    (
                        (mu - value) / nu
                        + 0.5 * (sigma / nu) ** 2
                        + normal_lcdf(mu + (sigma ** 2) / nu, sigma, value)
                    ),
                ),
                normal_lcdf(mu, sigma, value),
            ),
            0 < sigma,
            0 < nu,
        )

    def _distr_parameters_for_repr(self):
        return ["mu", "sigma", "nu"]


class VonMises(CircularContinuous):
    r"""
    Univariate VonMises log-likelihood.

    The pdf of this distribution is

    .. math::

        f(x \mid \mu, \kappa) =
            \frac{e^{\kappa\cos(x-\mu)}}{2\pi I_0(\kappa)}

    where :math:`I_0` is the modified Bessel function of order 0.

    .. plot::

        import matplotlib.pyplot as plt
        import numpy as np
        import scipy.stats as st
        import arviz as az
        plt.style.use('arviz-darkgrid')
        x = np.linspace(-np.pi, np.pi, 200)
        mus = [0., 0., 0.,  -2.5]
        kappas = [.01, 0.5,  4., 2.]
        for mu, kappa in zip(mus, kappas):
            pdf = st.vonmises.pdf(x, kappa, loc=mu)
            plt.plot(x, pdf, label=r'$\mu$ = {}, $\kappa$ = {}'.format(mu, kappa))
        plt.xlabel('x', fontsize=12)
        plt.ylabel('f(x)', fontsize=12)
        plt.legend(loc=1)
        plt.show()

    ========  ==========================================
    Support   :math:`x \in [-\pi, \pi]`
    Mean      :math:`\mu`
    Variance  :math:`1-\frac{I_1(\kappa)}{I_0(\kappa)}`
    ========  ==========================================

    Parameters
    ----------
    mu: float
        Mean.
    kappa: float
        Concentration (\frac{1}{kappa} is analogous to \sigma^2).
    """

    rv_op = vonmises

    @classmethod
    def dist(cls, mu=0.0, kappa=None, *args, **kwargs):
        mu = at.as_tensor_variable(floatX(mu))
        kappa = at.as_tensor_variable(floatX(kappa))
        assert_negative_support(kappa, "kappa", "VonMises")
        return super().dist([mu, kappa], *args, **kwargs)

    def logp(value, mu, kappa):
        """
        Calculate log-probability of VonMises distribution at specified value.

        Parameters
        ----------
        value: numeric
            Value(s) for which log-probability is calculated. If the log probabilities for multiple
            values are desired the values must be provided in a numpy array or Aesara tensor

        Returns
        -------
        TensorVariable
        """
        return bound(
            kappa * at.cos(mu - value) - (at.log(2 * np.pi) + log_i0(kappa)),
            kappa > 0,
            value >= -np.pi,
            value <= np.pi,
        )


class SkewNormalRV(RandomVariable):
    name = "skewnormal"
    ndim_supp = 0
    ndims_params = [0, 0, 0]
    dtype = "floatX"
    _print_name = ("SkewNormal", "\\operatorname{SkewNormal}")

    @classmethod
    def rng_fn(cls, rng, mu, sigma, alpha, size=None):
        return stats.skewnorm.rvs(a=alpha, loc=mu, scale=sigma, size=size, random_state=rng)


skewnormal = SkewNormalRV()


class SkewNormal(Continuous):
    r"""
    Univariate skew-normal log-likelihood.

     The pdf of this distribution is

    .. math::

       f(x \mid \mu, \tau, \alpha) =
       2 \Phi((x-\mu)\sqrt{\tau}\alpha) \phi(x,\mu,\tau)

    .. plot::

        import matplotlib.pyplot as plt
        import numpy as np
        import scipy.stats as st
        import arviz as az
        plt.style.use('arviz-darkgrid')
        x = np.linspace(-4, 4, 200)
        for alpha in [-6, 0, 6]:
            pdf = st.skewnorm.pdf(x, alpha, loc=0, scale=1)
            plt.plot(x, pdf, label=r'$\mu$ = {}, $\sigma$ = {}, $\alpha$ = {}'.format(0, 1, alpha))
        plt.xlabel('x', fontsize=12)
        plt.ylabel('f(x)', fontsize=12)
        plt.legend(loc=1)
        plt.show()

    ========  ==========================================
    Support   :math:`x \in \mathbb{R}`
    Mean      :math:`\mu + \sigma \sqrt{\frac{2}{\pi}} \frac {\alpha }{{\sqrt {1+\alpha ^{2}}}}`
    Variance  :math:`\sigma^2 \left(  1-\frac{2\alpha^2}{(\alpha^2+1) \pi} \right)`
    ========  ==========================================

    Skew-normal distribution can be parameterized either in terms of precision
    or standard deviation. The link between the two parametrizations is
    given by

    .. math::
       \tau = \dfrac{1}{\sigma^2}

    Parameters
    ----------
    mu: float
        Location parameter.
    sigma: float
        Scale parameter (sigma > 0).
    tau: float
        Alternative scale parameter (tau > 0).
    alpha: float
        Skewness parameter.

    Notes
    -----
    When alpha=0 we recover the Normal distribution and mu becomes the mean,
    tau the precision and sigma the standard deviation. In the limit of alpha
    approaching plus/minus infinite we get a half-normal distribution.

    """
    rv_op = skewnormal

    @classmethod
    def dist(cls, alpha=1, mu=0.0, sigma=None, tau=None, sd=None, *args, **kwargs):
        if sd is not None:
            sigma = sd

        tau, sigma = get_tau_sigma(tau=tau, sigma=sigma)
        alpha = at.as_tensor_variable(floatX(alpha))
        mu = at.as_tensor_variable(floatX(mu))
        tau = at.as_tensor_variable(tau)
        sigma = at.as_tensor_variable(sigma)

        assert_negative_support(tau, "tau", "SkewNormal")
        assert_negative_support(sigma, "sigma", "SkewNormal")

        return super().dist([mu, sigma, alpha], *args, **kwargs)

    def logp(value, mu, sigma, alpha):
        """
        Calculate log-probability of SkewNormal distribution at specified value.

        Parameters
        ----------
        value: numeric
            Value(s) for which log-probability is calculated. If the log probabilities for multiple
            values are desired the values must be provided in a numpy array or Aesara tensor

        Returns
        -------
        TensorVariable
        """
        tau, sigma = get_tau_sigma(sigma=sigma)
        return bound(
            at.log(1 + at.erf(((value - mu) * at.sqrt(tau) * alpha) / at.sqrt(2)))
            + (-tau * (value - mu) ** 2 + at.log(tau / np.pi / 2.0)) / 2.0,
            tau > 0,
            sigma > 0,
        )


class Triangular(BoundedContinuous):
    r"""
    Continuous Triangular log-likelihood

    The pdf of this distribution is

    .. math::

       \begin{cases}
         0 & \text{for } x < a, \\
         \frac{2(x-a)}{(b-a)(c-a)} & \text{for } a \le x < c, \\[4pt]
         \frac{2}{b-a}             & \text{for } x = c, \\[4pt]
         \frac{2(b-x)}{(b-a)(b-c)} & \text{for } c < x \le b, \\[4pt]
         0 & \text{for } b < x.
        \end{cases}

    .. plot::

        import matplotlib.pyplot as plt
        import numpy as np
        import scipy.stats as st
        import arviz as az
        plt.style.use('arviz-darkgrid')
        x = np.linspace(-2, 10, 500)
        lowers = [0., -1, 2]
        cs = [2., 0., 6.5]
        uppers = [4., 1, 8]
        for lower, c, upper in zip(lowers, cs, uppers):
            scale = upper - lower
            c_ = (c - lower) / scale
            pdf = st.triang.pdf(x, loc=lower, c=c_, scale=scale)
            plt.plot(x, pdf, label='lower = {}, c = {}, upper = {}'.format(lower,
                                                                           c,
                                                                           upper))
        plt.xlabel('x', fontsize=12)
        plt.ylabel('f(x)', fontsize=12)
        plt.legend(loc=1)
        plt.show()

    ========  ============================================================================
    Support   :math:`x \in [lower, upper]`
    Mean      :math:`\dfrac{lower + upper + c}{3}`
    Variance  :math:`\dfrac{upper^2 + lower^2 +c^2 - lower*upper - lower*c - upper*c}{18}`
    ========  ============================================================================

    Parameters
    ----------
    lower: float
        Lower limit.
    c: float
        mode
    upper: float
        Upper limit.
    """

    rv_op = triangular
    bound_args_indices = (0, 2)  # lower, upper

    @classmethod
    def dist(cls, lower=0, upper=1, c=0.5, *args, **kwargs):
        lower = at.as_tensor_variable(floatX(lower))
        upper = at.as_tensor_variable(floatX(upper))
        c = at.as_tensor_variable(floatX(c))

        return super().dist([lower, c, upper], *args, **kwargs)

    def logp(value, lower, c, upper):
        """
        Calculate log-probability of Triangular distribution at specified value.

        Parameters
        ----------
        value: numeric
            Value(s) for which log-probability is calculated. If the log probabilities for multiple
            values are desired the values must be provided in a numpy array or Aesara tensor

        Returns
        -------
        TensorVariable
        """
        return bound(
            at.switch(
                at.lt(value, c),
                at.log(2 * (value - lower) / ((upper - lower) * (c - lower))),
                at.log(2 * (upper - value) / ((upper - lower) * (upper - c))),
            ),
            lower <= value,
            value <= upper,
            lower <= c,
            c <= upper,
        )

    def logcdf(value, lower, c, upper):
        """
        Compute the log of the cumulative distribution function for Triangular distribution
        at the specified value.

        Parameters
        ----------
        value: numeric or np.ndarray or aesara.tensor
            Value(s) for which log CDF is calculated. If the log CDF for multiple
            values are desired the values must be provided in a numpy array or Aesara tensor.

        Returns
        -------
        TensorVariable
        """
        return bound(
            at.switch(
                at.le(value, lower),
                -np.inf,
                at.switch(
                    at.le(value, c),
                    at.log(((value - lower) ** 2) / ((upper - lower) * (c - lower))),
                    at.switch(
                        at.lt(value, upper),
                        at.log1p(-((upper - value) ** 2) / ((upper - lower) * (upper - c))),
                        0,
                    ),
                ),
            ),
            lower <= c,
            c <= upper,
        )


class Gumbel(Continuous):
    r"""
        Univariate Gumbel log-likelihood

    The pdf of this distribution is

    .. math::

       f(x \mid \mu, \beta) = \frac{1}{\beta}e^{-(z + e^{-z})}

    where

    .. math::

        z = \frac{x - \mu}{\beta}.

    .. plot::

        import matplotlib.pyplot as plt
        import numpy as np
        import scipy.stats as st
        import arviz as az
        plt.style.use('arviz-darkgrid')
        x = np.linspace(-10, 20, 200)
        mus = [0., 4., -1.]
        betas = [2., 2., 4.]
        for mu, beta in zip(mus, betas):
            pdf = st.gumbel_r.pdf(x, loc=mu, scale=beta)
            plt.plot(x, pdf, label=r'$\mu$ = {}, $\beta$ = {}'.format(mu, beta))
        plt.xlabel('x', fontsize=12)
        plt.ylabel('f(x)', fontsize=12)
        plt.legend(loc=1)
        plt.show()


    ========  ==========================================
    Support   :math:`x \in \mathbb{R}`
    Mean      :math:`\mu + \beta\gamma`, where :math:`\gamma` is the Euler-Mascheroni constant
    Variance  :math:`\frac{\pi^2}{6} \beta^2`
    ========  ==========================================

    Parameters
    ----------
    mu: float
        Location parameter.
    beta: float
        Scale parameter (beta > 0).
    """
    rv_op = gumbel

    @classmethod
    def dist(
        cls, mu: float = None, beta: float = None, no_assert: bool = False, **kwargs
    ) -> RandomVariable:

        mu = at.as_tensor_variable(floatX(mu))
        beta = at.as_tensor_variable(floatX(beta))

        if not no_assert:
            assert_negative_support(beta, "beta", "Gumbel")

        return super().dist([mu, beta], **kwargs)

    def _distr_parameters_for_repr(self):
        return ["mu", "beta"]

    def logp(
        value: Union[float, np.ndarray, TensorVariable],
        mu: Union[float, np.ndarray, TensorVariable],
        beta: Union[float, np.ndarray, TensorVariable],
    ) -> TensorVariable:
        """
        Calculate log-probability of Gumbel distribution at specified value.

        Parameters
        ----------
        value: numeric
            Value(s) for which log-probability is calculated. If the log probabilities for multiple
            values are desired the values must be provided in a numpy array or Aesara tensor

        Returns
        -------
        TensorVariable
        """
        scaled = (value - mu) / beta
        return bound(
            -scaled - at.exp(-scaled) - at.log(beta),
            0 < beta,
        )

    def logcdf(
        value: Union[float, np.ndarray, TensorVariable],
        mu: Union[float, np.ndarray, TensorVariable],
        beta: Union[float, np.ndarray, TensorVariable],
    ) -> TensorVariable:
        """
        Compute the log of the cumulative distribution function for Gumbel distribution
        at the specified value.

        Parameters
        ----------
        value: numeric or np.ndarray or aesara.tensor
            Value(s) for which log CDF is calculated. If the log CDF for multiple
            values are desired the values must be provided in a numpy array or Aesara tensor.

        Returns
        -------
        TensorVariable
        """
        return bound(
            -at.exp(-(value - mu) / beta),
            0 < beta,
        )


class RiceRV(RandomVariable):
    name = "rice"
    ndim_supp = 0
    ndims_params = [0, 0]
    dtype = "floatX"
    _print_name = ("Rice", "\\operatorname{Rice}")

    @classmethod
    def rng_fn(cls, rng, b, sigma, size=None):
        return stats.rice.rvs(b=b, scale=sigma, size=size, random_state=rng)


rice = RiceRV()


class Rice(PositiveContinuous):
    r"""
    Rice distribution.

    .. math::

       f(x\mid \nu ,\sigma )=
       {\frac  {x}{\sigma ^{2}}}\exp
       \left({\frac  {-(x^{2}+\nu ^{2})}{2\sigma ^{2}}}\right)I_{0}\left({\frac  {x\nu }{\sigma ^{2}}}\right),

    .. plot::

        import matplotlib.pyplot as plt
        import numpy as np
        import scipy.stats as st
        import arviz as az
        plt.style.use('arviz-darkgrid')
        x = np.linspace(0, 8, 500)
        nus = [0., 0., 4., 4.]
        sigmas = [1., 2., 1., 2.]
        for nu, sigma in  zip(nus, sigmas):
            pdf = st.rice.pdf(x, nu / sigma, scale=sigma)
            plt.plot(x, pdf, label=r'$\nu$ = {}, $\sigma$ = {}'.format(nu, sigma))
        plt.xlabel('x', fontsize=12)
        plt.ylabel('f(x)', fontsize=12)
        plt.legend(loc=1)
        plt.show()

    ========  ==============================================================
    Support   :math:`x \in (0, \infty)`
    Mean      :math:`\sigma {\sqrt  {\pi /2}}\,\,L_{{1/2}}(-\nu ^{2}/2\sigma ^{2})`
    Variance  :math:`2\sigma ^{2}+\nu ^{2}-{\frac  {\pi \sigma ^{2}}{2}}L_{{1/2}}^{2}\left({\frac  {-\nu ^{2}}{2\sigma ^{2}}}\right)`
    ========  ==============================================================


    Parameters
    ----------
    nu: float
        noncentrality parameter.
    sigma: float
        scale parameter.
    b: float
        shape parameter (alternative to nu).

    Notes
    -----
    The distribution :math:`\mathrm{Rice}\left(|\nu|,\sigma\right)` is the
    distribution of :math:`R=\sqrt{X^2+Y^2}` where :math:`X\sim N(\nu \cos{\theta}, \sigma^2)`,
    :math:`Y\sim N(\nu \sin{\theta}, \sigma^2)` are independent and for any
    real :math:`\theta`.

    The distribution is defined with either nu or b.
    The link between the two parametrizations is given by

    .. math::

       b = \dfrac{\nu}{\sigma}

    """
    rv_op = rice

    @classmethod
    def dist(cls, nu=None, sigma=None, b=None, sd=None, *args, **kwargs):
        if sd is not None:
            sigma = sd

        nu, b, sigma = cls.get_nu_b(nu, b, sigma)
        b = at.as_tensor_variable(floatX(b))
        sigma = at.as_tensor_variable(floatX(sigma))

        return super().dist([b, sigma], *args, **kwargs)

    @classmethod
    def get_nu_b(cls, nu, b, sigma):
        if sigma is None:
            sigma = 1.0
        if nu is None and b is not None:
            nu = b * sigma
            return nu, b, sigma
        elif nu is not None and b is None:
            b = nu / sigma
            return nu, b, sigma
        raise ValueError("Rice distribution must specify either nu" " or b.")

    def logp(value, b, sigma):
        """
        Calculate log-probability of Rice distribution at specified value.

        Parameters
        ----------
        value: numeric
            Value(s) for which log-probability is calculated. If the log probabilities for multiple
            values are desired the values must be provided in a numpy array or Aesara tensor

        Returns
        -------
        TensorVariable
        """
        x = value / sigma
        return bound(
            at.log(x * at.exp((-(x - b) * (x - b)) / 2) * i0e(x * b) / sigma),
            sigma >= 0,
            value > 0,
        )


class Logistic(Continuous):
    r"""
    Logistic log-likelihood.

    The pdf of this distribution is

    .. math::

       f(x \mid \mu, s) =
           \frac{\exp\left(-\frac{x - \mu}{s}\right)}{s \left(1 + \exp\left(-\frac{x - \mu}{s}\right)\right)^2}

    .. plot::

        import matplotlib.pyplot as plt
        import numpy as np
        import scipy.stats as st
        import arviz as az
        plt.style.use('arviz-darkgrid')
        x = np.linspace(-5, 5, 200)
        mus = [0., 0., 0., -2.]
        ss = [.4, 1., 2., .4]
        for mu, s in zip(mus, ss):
            pdf = st.logistic.pdf(x, loc=mu, scale=s)
            plt.plot(x, pdf, label=r'$\mu$ = {}, $s$ = {}'.format(mu, s))
        plt.xlabel('x', fontsize=12)
        plt.ylabel('f(x)', fontsize=12)
        plt.legend(loc=1)
        plt.show()

    ========  ==========================================
    Support   :math:`x \in \mathbb{R}`
    Mean      :math:`\mu`
    Variance  :math:`\frac{s^2 \pi^2}{3}`
    ========  ==========================================


    Parameters
    ----------
    mu: float
        Mean.
    s: float
        Scale (s > 0).
    """

    rv_op = logistic

    @classmethod
    def dist(cls, mu=0.0, s=1.0, *args, **kwargs):
        mu = at.as_tensor_variable(floatX(mu))
        s = at.as_tensor_variable(floatX(s))
        return super().dist([mu, s], *args, **kwargs)

    def logp(value, mu, s):
        """
        Calculate log-probability of Logistic distribution at specified value.

        Parameters
        ----------
        value: numeric
            Value(s) for which log-probability is calculated. If the log probabilities for multiple
            values are desired the values must be provided in a numpy array or Aesara tensor

        Returns
        -------
        TensorVariable
        """

        return bound(
            -(value - mu) / s - at.log(s) - 2 * at.log1p(at.exp(-(value - mu) / s)),
            s > 0,
        )

    def logcdf(value, mu, s):
        r"""
        Compute the log of the cumulative distribution function for Logistic distribution
        at the specified value.

        Parameters
        ----------
        value: numeric or np.ndarray or aesara.tensor
            Value(s) for which log CDF is calculated. If the log CDF for multiple
            values are desired the values must be provided in a numpy array or Aesara tensor.

        Returns
        -------
        TensorVariable
        """

        return bound(
            -at.log1pexp(-(value - mu) / s),
            0 < s,
        )


class LogitNormalRV(RandomVariable):
    name = "logit_normal"
    ndim_supp = 0
    ndims_params = [0, 0]
    dtype = "floatX"
    _print_name = ("logitNormal", "\\operatorname{logitNormal}")

    @classmethod
    def rng_fn(cls, rng, mu, sigma, size=None):
        return expit(stats.norm.rvs(loc=mu, scale=sigma, size=size, random_state=rng))


logit_normal = LogitNormalRV()


class LogitNormal(UnitContinuous):
    r"""
    Logit-Normal log-likelihood.

    The pdf of this distribution is

    .. math::
       f(x \mid \mu, \tau) =
           \frac{1}{x(1-x)} \sqrt{\frac{\tau}{2\pi}}
           \exp\left\{ -\frac{\tau}{2} (logit(x)-\mu)^2 \right\}


    .. plot::

        import matplotlib.pyplot as plt
        import numpy as np
        import scipy.stats as st
        from scipy.special import logit
        import arviz as az
        plt.style.use('arviz-darkgrid')
        x = np.linspace(0.0001, 0.9999, 500)
        mus = [0., 0., 0., 1.]
        sigmas = [0.3, 1., 2., 1.]
        for mu, sigma in  zip(mus, sigmas):
            pdf = st.norm.pdf(logit(x), loc=mu, scale=sigma) * 1/(x * (1-x))
            plt.plot(x, pdf, label=r'$\mu$ = {}, $\sigma$ = {}'.format(mu, sigma))
            plt.legend(loc=1)
        plt.show()

    ========  ==========================================
    Support   :math:`x \in (0, 1)`
    Mean      no analytical solution
    Variance  no analytical solution
    ========  ==========================================

    Parameters
    ----------
    mu: float
        Location parameter.
    sigma: float
        Scale parameter (sigma > 0).
    tau: float
        Scale parameter (tau > 0).
    """
    rv_op = logit_normal

    @classmethod
    def dist(cls, mu=0, sigma=None, tau=None, sd=None, **kwargs):
        if sd is not None:
            sigma = sd
        mu = at.as_tensor_variable(floatX(mu))
        tau, sigma = get_tau_sigma(tau=tau, sigma=sigma)
        sigma = sd = at.as_tensor_variable(sigma)
        tau = at.as_tensor_variable(tau)
        assert_negative_support(sigma, "sigma", "LogitNormal")
        assert_negative_support(tau, "tau", "LogitNormal")

        return super().dist([mu, sigma], **kwargs)

    def logp(value, mu, sigma):
        """
        Calculate log-probability of LogitNormal distribution at specified value.

        Parameters
        ----------
        value: numeric
            Value(s) for which log-probability is calculated. If the log probabilities for multiple
            values are desired the values must be provided in a numpy array or Aesara tensor

        Returns
        -------
        TensorVariable
        """
        tau, sigma = get_tau_sigma(sigma=sigma)
        return bound(
            -0.5 * tau * (logit(value) - mu) ** 2
            + 0.5 * at.log(tau / (2.0 * np.pi))
            - at.log(value * (1 - value)),
            value > 0,
            value < 1,
            tau > 0,
        )


def _interpolated_argcdf(p, pdf, cdf, x):
    index = np.searchsorted(cdf, p) - 1
    slope = (pdf[index + 1] - pdf[index]) / (x[index + 1] - x[index])

    return x[index] + np.where(
        np.abs(slope) <= 1e-8,
        np.where(np.abs(pdf[index]) <= 1e-8, np.zeros(index.shape), (p - cdf[index]) / pdf[index]),
        (-pdf[index] + np.sqrt(pdf[index] ** 2 + 2 * slope * (p - cdf[index]))) / slope,
    )


class InterpolatedRV(RandomVariable):
    name = "interpolated"
    ndim_supp = 0
    ndims_params = [1, 1, 1]
    dtype = "floatX"
    _print_name = ("Interpolated", "\\operatorname{Interpolated}")

    @classmethod
    def rng_fn(cls, rng, x, pdf, cdf, size=None):
        p = rng.uniform(size=size)
        return _interpolated_argcdf(p, pdf, cdf, x)


interpolated = InterpolatedRV()


class Interpolated(BoundedContinuous):
    r"""
    Univariate probability distribution defined as a linear interpolation
    of probability density function evaluated on some lattice of points.

    The lattice can be uneven, so the steps between different points can have
    different size and it is possible to vary the precision between regions
    of the support.

    The probability density function values don not have to be normalized, as the
    interpolated density is any way normalized to make the total probability
    equal to $1$.

    Both parameters ``x_points`` and values ``pdf_points`` are not variables, but
    plain array-like objects, so they are constant and cannot be sampled.

    .. plot::

        import matplotlib.pyplot as plt
        import numpy as np
        import pymc3 as pm
        import arviz as az
        from scipy.stats import gamma
        plt.style.use('arviz-darkgrid')
        rv = gamma(1.99)
        x = np.linspace(rv.ppf(0.01),rv.ppf(0.99), 1000)
        points = np.linspace(x[0], x[-1], 50)
        pdf = rv.pdf(points)
        interpolated = pm.Interpolated.dist(points, pdf)
        fig, ax = plt.subplots(1, 1)
        ax.plot(x, rv.pdf(x), 'C0', linestyle = '--',  label='Original Gamma pdf',alpha=0.8,lw=2)
        ax.plot(points, pdf, color='black', marker='o', label='Lattice Points',alpha=0.5,linestyle='')
        ax.plot(x, np.exp(interpolated.logp(x).eval()),'C1',label='Interpolated pdf',alpha=0.8,lw=3)
        r = interpolated.random(size=1000)
        ax.hist(r, density=True, alpha=0.4,align ='mid',color='grey')
        ax.legend(loc='best', frameon=False)
        plt.show()

    ========  ===========================================
    Support   :math:`x \in [x\_points[0], x\_points[-1]]`
    ========  ===========================================

    Parameters
    ----------
    x_points: array-like
        A monotonically growing list of values. Must be non-symbolic
    pdf_points: array-like
        Probability density function evaluated on lattice ``x_points``. Must
        be non-symbolic
    """

    rv_op = interpolated

    @classmethod
    def dist(cls, x_points, pdf_points, *args, **kwargs):

        interp = InterpolatedUnivariateSpline(x_points, pdf_points, k=1, ext="zeros")

        Z = interp.integral(x_points[0], x_points[-1])
        cdf_points = interp.antiderivative()(x_points) / Z
        pdf_points = pdf_points / Z

        x_points = at.constant(floatX(x_points))
        pdf_points = at.constant(floatX(pdf_points))
        cdf_points = at.constant(floatX(cdf_points))

        # lower = at.as_tensor_variable(x_points[0])
        # upper = at.as_tensor_variable(x_points[-1])
        # median = _interpolated_argcdf(0.5, pdf_points, cdf_points, x_points)

        return super().dist([x_points, pdf_points, cdf_points], **kwargs)

    def logp(value, x_points, pdf_points, cdf_points):
        """
        Calculate log-probability of Interpolated distribution at specified value.

        Parameters
        ----------
        value: numeric
            Value(s) for which log-probability is calculated. If the log probabilities for multiple
            values are desired the values must be provided in a numpy array or Aesara tensor

        Returns
        -------
        TensorVariable
        """
        # x_points and pdf_points are expected to be non-symbolic arrays wrapped
        # within a tensor.constant. We use the .data method to retrieve them
        interp = InterpolatedUnivariateSpline(x_points.data, pdf_points.data, k=1, ext="zeros")
        Z = interp.integral(x_points.data[0], x_points.data[-1])

        # interp and Z are converted to symbolic variables here
        interp_op = SplineWrapper(interp)
        Z = at.constant(Z)

        return at.log(interp_op(value) / Z)

    def _distr_parameters_for_repr(self):
        return []


class MoyalRV(RandomVariable):
    name = "moyal"
    ndim_supp = 0
    ndims_params = [0, 0]
    dtype = "floatX"
    _print_name = ("Moyal", "\\operatorname{Moyal}")

    @classmethod
    def rng_fn(cls, rng, mu, sigma, size=None):
        return stats.moyal.rvs(mu, sigma, size=size, random_state=rng)


moyal = MoyalRV()


class Moyal(Continuous):
    r"""
    Moyal log-likelihood.

    The pdf of this distribution is

    .. math::

       f(x \mid \mu,\sigma) = \frac{1}{\sqrt{2\pi}\sigma}e^{-\frac{1}{2}\left(z + e^{-z}\right)},

    where

    .. math::

       z = \frac{x-\mu}{\sigma}.

    .. plot::

        import matplotlib.pyplot as plt
        import numpy as np
        import scipy.stats as st
        import arviz as az
        plt.style.use('arviz-darkgrid')
        x = np.linspace(-10, 20, 200)
        mus = [-1., 0., 4.]
        sigmas = [2., 2., 4.]
        for mu, sigma in zip(mus, sigmas):
            pdf = st.moyal.pdf(x, loc=mu, scale=sigma)
            plt.plot(x, pdf, label=r'$\mu$ = {}, $\sigma$ = {}'.format(mu, sigma))
        plt.xlabel('x', fontsize=12)
        plt.ylabel('f(x)', fontsize=12)
        plt.legend(loc=1)
        plt.show()

    ========  ==============================================================
    Support   :math:`x \in (-\infty, \infty)`
    Mean      :math:`\mu + \sigma\left(\gamma + \log 2\right)`, where :math:`\gamma` is the Euler-Mascheroni constant
    Variance  :math:`\frac{\pi^{2}}{2}\sigma^{2}`
    ========  ==============================================================

    Parameters
    ----------
    mu: float
        Location parameter.
    sigma: float
        Scale parameter (sigma > 0).
    """
    rv_op = moyal

    @classmethod
    def dist(cls, mu=0, sigma=1.0, *args, **kwargs):
        mu = at.as_tensor_variable(floatX(mu))
        sigma = at.as_tensor_variable(floatX(sigma))

        assert_negative_support(sigma, "sigma", "Moyal")

        return super().dist([mu, sigma], *args, **kwargs)

    def logp(value, mu, sigma):
        """
        Calculate log-probability of Moyal distribution at specified value.

        Parameters
        ----------
        value: numeric
            Value(s) for which log-probability is calculated. If the log probabilities for multiple
            values are desired the values must be provided in a numpy array or Aesara tensor

        Returns
        -------
        TensorVariable
        """
        scaled = (value - mu) / sigma
        return bound(
            (-(1 / 2) * (scaled + at.exp(-scaled)) - at.log(sigma) - (1 / 2) * at.log(2 * np.pi)),
            0 < sigma,
        )

    def logcdf(value, mu, sigma):
        """
        Compute the log of the cumulative distribution function for Moyal distribution
        at the specified value.

        Parameters
        ----------
        value: numeric or np.ndarray or aesara.tensor
            Value(s) for which log CDF is calculated. If the log CDF for multiple
            values are desired the values must be provided in a numpy array or Aesara tensor.

        Returns
        -------
        TensorVariable
        """
        scaled = (value - mu) / sigma
        return bound(
            at.log(at.erfc(at.exp(-scaled / 2) * (2 ** -0.5))),
            0 < sigma,
        )


class PolyaGammaRV(RandomVariable):
    """Polya-Gamma random variable."""

    name = "polyagamma"
    ndim_supp = 0
    ndims_params = [0, 0]
    dtype = "floatX"
    _print_name = ("PG", "\\operatorname{PG}")

    def __call__(self, h=1.0, z=0.0, size=None, **kwargs):
        return super().__call__(h, z, size=size, **kwargs)

    @classmethod
    def rng_fn(cls, rng, h, z, size=None):
        """
        Generate a random sample from the distribution with the given parameters

        Parameters
        ----------
        rng : {None, int, array_like[ints], SeedSequence, BitGenerator, Generator}
            A seed to initialize the random number generator. If None, then fresh,
            unpredictable entropy will be pulled from the OS. If an ``int`` or
            ``array_like[ints]`` is passed, then it will be passed to
            `SeedSequence` to derive the initial `BitGenerator` state. One may also
            pass in a `SeedSequence` instance.
            Additionally, when passed a `BitGenerator`, it will be wrapped by
            `Generator`. If passed a `Generator`, it will be returned unaltered.
        h : scalar or sequence
            The shape parameter of the distribution.
        z : scalar or sequence
            The exponential tilting parameter.
        size : int or tuple of ints, optional
            The number of elements to draw from the distribution. If size is
            ``None`` (default) then a single value is returned. If a tuple of
            integers is passed, the returned array will have the same shape.
            If the element(s) of size is not an integer type, it will be truncated
            to the largest integer smaller than its value (e.g (2.1, 1) -> (2, 1)).
            This parameter only applies if `h` and `z` are scalars.
        """
        # handle the kind of rng passed to the sampler
        bg = rng._bit_generator if isinstance(rng, np.random.RandomState) else rng
        return random_polyagamma(h, z, size=size, random_state=bg).astype(aesara.config.floatX)


polyagamma = PolyaGammaRV()


class _PolyaGammaLogDistFunc(Op):
    __props__ = ("get_pdf",)

    def __init__(self, get_pdf=False):
        self.get_pdf = get_pdf

    def make_node(self, x, h, z):
        x = at.as_tensor_variable(floatX(x))
        h = at.as_tensor_variable(floatX(h))
        z = at.as_tensor_variable(floatX(z))
        shape = broadcast_shape(x, h, z)
        broadcastable = [] if not shape else [False] * len(shape)
        return Apply(self, [x, h, z], [at.TensorType(aesara.config.floatX, broadcastable)()])

    def perform(self, node, ins, outs):
        x, h, z = ins[0], ins[1], ins[2]
        outs[0][0] = (
            polyagamma_pdf(x, h, z, return_log=True)
            if self.get_pdf
            else polyagamma_cdf(x, h, z, return_log=True)
        ).astype(aesara.config.floatX)


class PolyaGamma(PositiveContinuous):
    r"""
    The Polya-Gamma distribution.

    The distribution is parametrized by ``h`` (shape parameter) and ``z``
    (exponential tilting parameter). The pdf of this distribution is

    .. math::

       f(x \mid h, z) = cosh^h(\frac{z}{2})e^{-\frac{1}{2}xz^2}f(x \mid h, 0),
    where :math:`f(x \mid h, 0)` is the pdf of a :math:`PG(h, 0)` variable.
    Notice that the pdf of this distribution is expressed as an alternating-sign
    sum of inverse-Gaussian densities.

    .. math::

        X = \Sigma_{k=1}^{\infty}\frac{Ga(h, 1)}{d_k},

    where :math:`d_k = 2(k - 0.5)^2\pi^2 + z^2/2`, :math:`Ga(h, 1)` is a gamma
    random variable with shape  parameter ``h`` and scale parameter ``1``.

    .. plot::

        import matplotlib.pyplot as plt
        import numpy as np
        from polyagamma import polyagamma_pdf
        plt.style.use('seaborn-darkgrid')
        x = np.linspace(0.01, 5, 500);x.sort()
        hs = [1., 5., 10., 15.]
        zs = [0.] * 4
        for h, z in zip(hs, zs):
            pdf = polyagamma_pdf(x, h=h, z=z)
            plt.plot(x, pdf, label=r'$h$ = {}, $z$ = {}'.format(h, z))
        plt.xlabel('x', fontsize=12)
        plt.ylabel('f(x)', fontsize=12)
        plt.legend(loc=1)
        plt.show()

    ========  =============================
    Support   :math:`x \in (0, \infty)`
    Mean      :math:`dfrac{h}{4} if :math:`z=0`, :math:`\dfrac{tanh(z/2)h}{2z}` otherwise.
    Variance  :math:`0.041666688h` if :math:`z=0`, :math:`\dfrac{h(sinh(z) - z)(1 - tanh^2(z/2))}{4z^3}` otherwise.
    ========  =============================

    Parameters
    ----------
    h: float, optional
        The shape parameter of the distribution (h > 0).
    z: float, optional
        The exponential tilting parameter of the distribution.

    Examples
    --------
    .. code-block:: python

        rng = np.random.default_rng()
        with pm.Model():
            x = pm.PolyaGamma('x', h=1, z=5.5)
        with pm.Model():
            x = pm.PolyaGamma('x', h=25, z=-2.3, rng=rng, size=(100, 5))

    References
    ----------
    .. [1] Polson, Nicholas G., James G. Scott, and Jesse Windle.
           "Bayesian inference for logistic models using Pólya–Gamma latent
           variables." Journal of the American statistical Association
           108.504 (2013): 1339-1349.
    .. [2] Windle, Jesse, Nicholas G. Polson, and James G. Scott.
           "Sampling Polya-Gamma random variates: alternate and approximate
           techniques." arXiv preprint arXiv:1405.0506 (2014)
    .. [3] Luc Devroye. "On exact simulation algorithms for some distributions
           related to Jacobi theta functions." Statistics & Probability Letters,
           Volume 79, Issue 21, (2009): 2251-2259.
    .. [4] Windle, J. (2013). Forecasting high-dimensional, time-varying
           variance-covariance matrices with high-frequency data and sampling
           Pólya-Gamma random variates for posterior distributions derived
           from logistic likelihoods.(PhD thesis). Retrieved from
           http://hdl.handle.net/2152/21842
    """
    rv_op = polyagamma

    @classmethod
    def dist(cls, h=1.0, z=0.0, **kwargs):
        h = at.as_tensor_variable(floatX(h))
        z = at.as_tensor_variable(floatX(z))

        msg = f"The variable {h} specified for PolyaGamma has non-positive "
        msg += "values, making it unsuitable for this parameter."
        Assert(msg)(h, at.all(at.gt(h, 0.0)))

        return super().dist([h, z], **kwargs)

    def logp(value, h, z):
        """
        Calculate log-probability of Polya-Gamma distribution at specified value.

        Parameters
        ----------
        value: numeric
            Value(s) for which log-probability is calculated. If the log
            probabilities for multiple values are desired the values must be
            provided in a numpy array.

        Returns
        -------
        TensorVariable
        """

        return bound(_PolyaGammaLogDistFunc(True)(value, h, z), h > 0, value > 0)

    def logcdf(value, h, z):
        """
        Compute the log of the cumulative distribution function for the
        Polya-Gamma distribution at the specified value.

        Parameters
        ----------
        value: numeric or np.ndarray or `TensorVariable`
            Value(s) for which log CDF is calculated. If the log CDF for multiple
            values are desired the values must be provided in a numpy array.

        Returns
        -------
        TensorVariable
        """
        return bound(_PolyaGammaLogDistFunc(False)(value, h, z), h > 0, value > 0)
