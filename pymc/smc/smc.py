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
import abc

from abc import ABC
from typing import Dict

import aesara.tensor as at
import numpy as np

from aesara.graph.basic import clone_replace
from scipy.special import logsumexp
from scipy.stats import multivariate_normal

from pymc.aesaraf import (
    compile_rv_inplace,
    floatX,
    inputvars,
    join_nonshared_inputs,
    make_shared_replacements,
)
from pymc.backends.ndarray import NDArray
from pymc.blocking import DictToArrayBijection
from pymc.model import Point, modelcontext
from pymc.sampling import sample_prior_predictive
from pymc.step_methods.metropolis import MultivariateNormalProposal
from pymc.vartypes import discrete_types


class SMC_KERNEL(ABC):
    """Base class for the Sequential Monte Carlo kernels

    To creat a new SMC kernel you should subclass from this.

    Before sampling, the following methods are called once in order:

        initialize_population
            Choose initial population of SMC particles. Should return a dictionary
            with {var.name : numpy array of size (draws, var.size)}. Defaults
            to sampling from the prior distribution. This method is only called
            if `start` is not specified.

        _initialize_kernel: default
            Creates initial population of particles in the variable
            `self.tempered_posterior` and populates the `self.var_info` dictionary
            with information about model variables shape and size as
            {var.name : (var.shape, var.size)

            The functions self.prior_logp_func and self.likelihood_logp_func are
            created in this step. These expect a 1D numpy array with the summed
            sizes of each raveled model variable (in the order specified in
            model.inial_point).

            Finally, this method computes the log prior and log likelihood for
            the initial particles, and saves them in self.prior_logp and
            self.likelihood_logp.

            This method should not be modified.

        setup_kernel: optional
            May include any logic that should be performed before sampling
            starts.

    During each sampling stage the following methods are called in order:

        update_beta_and_weights: default
            The inverse temperature self.beta is updated based on the self.likelihood_logp
            and `threshold` parameter

            The importance self.weights of each particle are computed from the old and newly
            selected inverse temperature

            The iteration number stored in `self.iteration` is updated by this method.

            Finally the model log_marginal_likelihood of the tempered posterior
            is updated from these weights

        resample: default
            The particles in self.posterior are sampled with replacement based
            on self.weights, and the used resampling indexes are saved in
            `self.resampling_indexes`.

            The arrays self.prior_logp, self.likelihood_logp are rearranged according
            to the order of the resampled particles. self.tempered_posterior_logp
            is computed from these and the current self.beta

        tune: optional
            May include logic that should be performed before every mutation step

        mutate: REQUIRED
            Mutate particles in self.tempered_posterior

            This method is further responsible to update the self.prior_logp,
            self.likelihod_logp and self.tempered_posterior_logp, corresponding
            to each mutated particle

        sample_stats: default
            Returns important sampling_stats at the end of each stage in a dictionary
            format. This will be saved in the final InferenceData objcet under `sample_stats`.

    Finally, at the end of sampling the following methods are called:

        _posterior_to_trace: default
            Convert final population of particles to a posterior trace object.
            This method should not be modified.

        sample_settings: default:
            Returns important sample_settings at the end of sampling in a dictionary
            format. This will be saved in the final InferenceData objcet under `sample_stats`.

    """

    def __init__(
        self,
        draws=2000,
        start=None,
        model=None,
        random_seed=-1,
        threshold=0.5,
    ):
        """

        Parameters
        ----------
        draws: int
            The number of samples to draw from the posterior (i.e. last stage). And also the number of
            independent chains. Defaults to 2000.
        start: dict, or array of dict
            Starting point in parameter space. It should be a list of dict with length `chains`.
            When None (default) the starting point is sampled from the prior distribution.
        model: Model (optional if in ``with`` context)).
        threshold: float
            Determines the change of beta from stage to stage, i.e.indirectly the number of stages,
            the higher the value of `threshold` the higher the number of stages. Defaults to 0.5.
            It should be between 0 and 1.

        """

        self.draws = draws
        self.start = start
        self.threshold = threshold
        self.model = model
        self.random_seed = random_seed

        if self.random_seed != -1:
            np.random.seed(self.random_seed)

        self.model = modelcontext(model)
        self.variables = inputvars(self.model.value_vars)

        self.var_info = {}
        self.tempered_posterior = None
        self.prior_logp = None
        self.likelihood_logp = None
        self.tempered_posterior_logp = None
        self.prior_logp_func = None
        self.likelihood_logp_func = None
        self.log_marginal_likelihood = 0
        self.beta = 0
        self.iteration = 0
        self.resampling_indexes = None
        self.weights = np.ones(self.draws) / self.draws

    def initialize_population(self) -> Dict[str, NDArray]:
        """Create an initial population from the prior distribution"""

        return sample_prior_predictive(
            self.draws,
            var_names=[v.name for v in self.model.unobserved_value_vars],
            model=self.model,
        )

    def _initialize_kernel(self):
        """Create variables and logp function necessary to run kernel

        This method should not be overwritten. If needed, use `setup_kernel`
        instead.

        """
        # Create dictionary that stores original variables shape and size
        initial_point = self.model.initial_point
        for v in self.variables:
            self.var_info[v.name] = (initial_point[v.name].shape, initial_point[v.name].size)

        # Create particles bijection map
        if self.start:
            init_rnd = self.start
        else:
            init_rnd = self.initialize_population()

        population = []
        for i in range(self.draws):
            point = Point({v.name: init_rnd[v.name][i] for v in self.variables}, model=self.model)
            population.append(DictToArrayBijection.map(point).data)
        self.tempered_posterior = np.array(floatX(population))

        # Initialize prior and likelihood log probabilities
        shared = make_shared_replacements(initial_point, self.variables, self.model)

        self.prior_logp_func = _logp_forw(
            initial_point, [self.model.varlogpt], self.variables, shared
        )
        self.likelihood_logp_func = _logp_forw(
            initial_point, [self.model.datalogpt], self.variables, shared
        )

        priors = [self.prior_logp_func(sample) for sample in self.tempered_posterior]
        likelihoods = [self.likelihood_logp_func(sample) for sample in self.tempered_posterior]

        self.prior_logp = np.array(priors).squeeze()
        self.likelihood_logp = np.array(likelihoods).squeeze()

    def setup_kernel(self):
        """Setup logic performed once before sampling starts"""
        pass

    def update_beta_and_weights(self):
        """Calculate the next inverse temperature (beta)

        The importance weights based on two sucesive tempered likelihoods (i.e.
        two successive values of beta) and updates the marginal likelihood estimate.
        """
        self.iteration += 1

        low_beta = old_beta = self.beta
        up_beta = 2.0
        rN = int(len(self.likelihood_logp) * self.threshold)

        while up_beta - low_beta > 1e-6:
            new_beta = (low_beta + up_beta) / 2.0
            log_weights_un = (new_beta - old_beta) * self.likelihood_logp
            log_weights = log_weights_un - logsumexp(log_weights_un)
            ESS = int(np.exp(-logsumexp(log_weights * 2)))
            if ESS == rN:
                break
            elif ESS < rN:
                up_beta = new_beta
            else:
                low_beta = new_beta
        if new_beta >= 1:
            new_beta = 1
            log_weights_un = (new_beta - old_beta) * self.likelihood_logp
            log_weights = log_weights_un - logsumexp(log_weights_un)

        self.beta = new_beta
        self.weights = np.exp(log_weights)
        # We normalize again to correct for small numerical errors that might build up
        self.weights /= self.weights.sum()
        self.log_marginal_likelihood += logsumexp(log_weights_un) - np.log(self.draws)

    def resample(self):
        """Resample particles based on importance weights"""
        self.resampling_indexes = np.random.choice(
            np.arange(self.draws), size=self.draws, p=self.weights
        )

        self.tempered_posterior = self.tempered_posterior[self.resampling_indexes]
        self.prior_logp = self.prior_logp[self.resampling_indexes]
        self.likelihood_logp = self.likelihood_logp[self.resampling_indexes]
        self.tempered_posterior_logp = self.prior_logp + self.likelihood_logp * self.beta

    def tune(self):
        """Tuning logic performed before every mutation step"""
        pass

    @abc.abstractmethod
    def mutate(self):
        """Apply kernel-specific perturbation to the particles once per stage"""
        pass

    def sample_stats(self) -> Dict:
        """Stats to be saved at the end of each stage

        These stats will be saved under `sample_stats` in the final InferenceData.
        """
        return {
            "log_marginal_likelihood": self.log_marginal_likelihood if self.beta == 1 else np.nan,
            "beta": self.beta,
        }

    def sample_settings(self) -> Dict:
        """Kernel settings to be saved once at the end of sampling

        These stats will be saved under `sample_stats` in the final InferenceData.

        """
        return {
            "_n_draws": self.draws,  # Default property name used in `SamplerReport`
            "threshold": self.threshold,
        }

    def _posterior_to_trace(self, chain=0) -> NDArray:
        """Save results into a PyMC trace

        This method shoud not be overwritten.
        """
        lenght_pos = len(self.tempered_posterior)
        varnames = [v.name for v in self.variables]

        with self.model:
            strace = NDArray(name=self.model.name)
            strace.setup(lenght_pos, chain)
        for i in range(lenght_pos):
            value = []
            size = 0
            for varname in varnames:
                shape, new_size = self.var_info[varname]
                var_samples = self.tempered_posterior[i][size : size + new_size]
                # Round discrete variable samples. The rounded values were the ones
                # actually used in the logp evaluations (see logp_forw)
                var = self.model[varname]
                if var.dtype in discrete_types:
                    var_samples = np.round(var_samples).astype(var.dtype)
                value.append(var_samples.reshape(shape))
                size += new_size
            strace.record(point={k: v for k, v in zip(varnames, value)})
        return strace


class IMH(SMC_KERNEL):
    """Independent Metropolis-Hastings SMC kernel"""

    def __init__(self, *args, n_steps=25, tune_steps=True, p_acc_rate=0.85, **kwargs):
        """
        Parameters
        ----------
        n_steps: int
            The number of steps of each Markov Chain. If ``tune_steps == True`` ``n_steps`` will be used
            for the first stage and for the others it will be determined automatically based on the
            acceptance rate and `p_acc_rate`, the max number of steps is ``n_steps``.
        tune_steps: bool
            Whether to compute the number of steps automatically or not. Defaults to True
        p_acc_rate: float
            Used to compute ``n_steps`` when ``tune_steps == True``. The higher the value of
            ``p_acc_rate`` the higher the number of steps computed automatically. Defaults to 0.85.
            It should be between 0 and 1.
        """
        super().__init__(*args, **kwargs)
        self.n_steps = n_steps
        self.tune_steps = tune_steps
        self.p_acc_rate = p_acc_rate

        self.max_steps = n_steps
        self.proposed = self.draws * self.n_steps
        self.proposal_dist = None
        self.acc_rate = None

    def tune(self):
        # Tune n_steps based on the acceptance rate (skip in first iteration)
        if self.tune_steps and self.iteration > 1:
            acc_rate = max(1.0 / self.proposed, self.acc_rate)
            self.n_steps = min(
                self.max_steps,
                max(2, int(np.log(1 - self.p_acc_rate) / np.log(1 - acc_rate))),
            )
            self.proposed = self.draws * self.n_steps

        # Update MVNormal proposal based on the mean and covariance of the
        # tempered posterior.
        cov = np.cov(self.tempered_posterior, ddof=0, rowvar=0)
        cov = np.atleast_2d(cov)
        cov += 1e-6 * np.eye(cov.shape[0])
        if np.isnan(cov).any() or np.isinf(cov).any():
            raise ValueError('Sample covariances not valid! Likely "draws" is too small!')
        mean = np.average(self.tempered_posterior, axis=0)
        self.proposal_dist = multivariate_normal(mean, cov)

    def mutate(self):
        """Independent Metropolis-Hastings perturbation."""
        ac_ = np.empty((self.n_steps, self.draws))

        cov = self.proposal_dist.cov
        log_R = np.log(np.random.rand(self.n_steps, self.draws))
        for n_step in range(self.n_steps):
            # The proposal is independent from the current point.
            # We have to take that into account to compute the Metropolis-Hastings acceptance
            proposal = floatX(self.proposal_dist.rvs(size=self.draws))
            proposal = proposal.reshape(len(proposal), -1)
            # To do that we compute the logp of moving to a new point
            forward = self.proposal_dist.logpdf(proposal)
            # And to going back from that new point
            backward = multivariate_normal(proposal.mean(axis=0), cov).logpdf(
                self.tempered_posterior
            )
            ll = np.array([self.likelihood_logp_func(prop) for prop in proposal])
            pl = np.array([self.prior_logp_func(prop) for prop in proposal])
            proposal_logp = pl + ll * self.beta
            accepted = log_R[n_step] < (
                (proposal_logp + backward) - (self.tempered_posterior_logp + forward)
            )
            ac_[n_step] = accepted
            self.tempered_posterior[accepted] = proposal[accepted]
            self.tempered_posterior_logp[accepted] = proposal_logp[accepted]
            self.prior_logp[accepted] = pl[accepted]
            self.likelihood_logp[accepted] = ll[accepted]

        self.acc_rate = np.mean(ac_)

    def sample_stats(self):
        stats = super().sample_stats()
        stats.update(
            {
                "accept_rate": self.acc_rate,
            }
        )
        return stats

    def sample_settings(self):
        stats = super().sample_settings()
        stats.update(
            {
                "_n_tune": self.n_steps,  # Default property name used in `SamplerReport`
                "tune_steps": self.tune_steps,
                "p_acc_rate": self.p_acc_rate,
            }
        )
        return stats


class MH(SMC_KERNEL):
    """Metropolis-Hastings SMC kernel"""

    def __init__(self, *args, n_steps=25, tune_steps=True, p_acc_rate=0.85, **kwargs):
        """
        Parameters
        ----------
        n_steps: int
            The number of steps of each Markov Chain.
        tune_steps: bool
            Whether to compute the number of steps automatically or not. Defaults to True
        p_acc_rate: float
            Used to compute ``n_steps`` when ``tune_steps == True``. The higher the value of
            ``p_acc_rate`` the higher the number of steps computed automatically. Defaults to 0.85.
            It should be between 0 and 1.
        """
        super().__init__(*args, **kwargs)
        self.n_steps = n_steps
        self.tune_steps = tune_steps
        self.p_acc_rate = p_acc_rate

        self.max_steps = n_steps
        self.proposed = self.draws * self.n_steps
        self.proposal_dist = None
        self.proposal_scales = None
        self.chain_acc_rate = None

    def setup_kernel(self):
        """Proposal dist is just a Multivariate Normal with unit identity covariance.
        Dimension specific scaling is provided by self.proposal_scales and set in self.tune()
        """
        ndim = self.tempered_posterior.shape[1]
        self.proposal_scales = np.full(self.draws, min(1, 2.38 ** 2 / ndim))

    def resample(self):
        super().resample()
        if self.iteration > 1:
            self.proposal_scales = self.proposal_scales[self.resampling_indexes]
            self.chain_acc_rate = self.chain_acc_rate[self.resampling_indexes]

    def tune(self):
        """Update proposal scales for each particle dimension and update number of MH steps"""
        if self.iteration > 1:
            # Rescale based on distance to 0.234 acceptance rate
            chain_scales = np.exp(np.log(self.proposal_scales) + (self.chain_acc_rate - 0.234))
            # Interpolate between individual and population scales
            self.proposal_scales = 0.5 * (chain_scales + chain_scales.mean())

            if self.tune_steps:
                acc_rate = max(1.0 / self.proposed, self.chain_acc_rate.mean())
                self.n_steps = min(
                    self.max_steps,
                    max(2, int(np.log(1 - self.p_acc_rate) / np.log(1 - acc_rate))),
                )
            self.proposed = self.draws * self.n_steps

        # Update MVNormal proposal based on the covariance of the tempered posterior.
        cov = np.cov(self.tempered_posterior, ddof=0, rowvar=0)
        cov = np.atleast_2d(cov)
        cov += 1e-6 * np.eye(cov.shape[0])
        if np.isnan(cov).any() or np.isinf(cov).any():
            raise ValueError('Sample covariances not valid! Likely "draws" is too small!')
        self.proposal_dist = MultivariateNormalProposal(cov)

    def mutate(self):
        """Metropolis-Hastings perturbation."""
        ac_ = np.empty((self.n_steps, self.draws))

        log_R = np.log(np.random.rand(self.n_steps, self.draws))
        for n_step in range(self.n_steps):
            proposal = floatX(
                self.tempered_posterior
                + self.proposal_dist(num_draws=self.draws) * self.proposal_scales[:, None]
            )
            ll = np.array([self.likelihood_logp_func(prop) for prop in proposal])
            pl = np.array([self.prior_logp_func(prop) for prop in proposal])

            proposal_logp = pl + ll * self.beta
            accepted = log_R[n_step] < (proposal_logp - self.tempered_posterior_logp)

            ac_[n_step] = accepted
            self.tempered_posterior[accepted] = proposal[accepted]
            self.prior_logp[accepted] = pl[accepted]
            self.likelihood_logp[accepted] = ll[accepted]
            self.tempered_posterior_logp[accepted] = proposal_logp[accepted]

        self.chain_acc_rate = np.mean(ac_, axis=0)

    def sample_stats(self):
        stats = super().sample_stats()
        stats.update(
            {
                "mean_accept_rate": self.chain_acc_rate.mean(),
                "mean_proposal_scale": self.proposal_scales.mean(),
            }
        )
        return stats

    def sample_settings(self):
        stats = super().sample_settings()
        stats.update(
            {
                "_n_tune": self.n_steps,  # Default property name used in `SamplerReport`
                "tune_steps": self.tune_steps,
                "p_acc_rate": self.p_acc_rate,
            }
        )
        return stats


def _logp_forw(point, out_vars, vars, shared):
    """Compile Aesara function of the model and the input and output variables.

    Parameters
    ----------
    out_vars: List
        containing :class:`pymc.Distribution` for the output variables
    vars: List
        containing :class:`pymc.Distribution` for the input variables
    shared: List
        containing :class:`aesara.tensor.Tensor` for depended shared data
    """

    # Convert expected input of discrete variables to (rounded) floats
    if any(var.dtype in discrete_types for var in vars):
        replace_int_to_float = {}
        replace_float_to_round = {}
        new_vars = []
        for var in vars:
            if var.dtype in discrete_types:
                float_var = at.TensorType("floatX", var.broadcastable)(var.name)
                replace_int_to_float[var] = float_var
                new_vars.append(float_var)

                round_float_var = at.round(float_var)
                round_float_var.name = var.name
                replace_float_to_round[float_var] = round_float_var
            else:
                new_vars.append(var)

        replace_int_to_float.update(shared)
        replace_float_to_round.update(shared)
        out_vars = clone_replace(out_vars, replace_int_to_float, strict=False)
        out_vars = clone_replace(out_vars, replace_float_to_round)
        vars = new_vars

    out_list, inarray0 = join_nonshared_inputs(point, out_vars, vars, shared)
    f = compile_rv_inplace([inarray0], out_list[0])
    f.trust_input = True
    return f
