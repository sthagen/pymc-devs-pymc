#   Copyright 2022- The PyMC Developers
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
#
#   MIT License
#
#   Copyright (c) 2021-2022 aesara-devs
#
#   Permission is hereby granted, free of charge, to any person obtaining a copy
#   of this software and associated documentation files (the "Software"), to deal
#   in the Software without restriction, including without limitation the rights
#   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#   copies of the Software, and to permit persons to whom the Software is
#   furnished to do so, subject to the following conditions:
#
#   The above copyright notice and this permission notice shall be included in all
#   copies or substantial portions of the Software.
#
#   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#   SOFTWARE.

import numpy as np

from aesara import tensor as at
from aesara.graph.basic import walk
from aesara.graph.op import HasInnerGraph
from scipy import stats as stats

from pymc.logprob.abstract import MeasurableVariable, icdf, logcdf, logprob


def assert_no_rvs(var):
    """Assert that there are no `MeasurableVariable` nodes in a graph."""

    def expand(r):
        owner = r.owner
        if owner:
            inputs = list(reversed(owner.inputs))

            if isinstance(owner.op, HasInnerGraph):
                inputs += owner.op.inner_outputs

            return inputs

    for v in walk([var], expand, False):
        if v.owner and isinstance(v.owner.op, MeasurableVariable):
            raise AssertionError(f"Variable {v} is a MeasurableVariable")


def simulate_poiszero_hmm(
    N, mu=10.0, pi_0_a=np.r_[1, 1], p_0_a=np.r_[5, 1], p_1_a=np.r_[1, 1], seed=None
):
    rng = np.random.default_rng(seed)

    p_0 = rng.dirichlet(p_0_a)
    p_1 = rng.dirichlet(p_1_a)

    Gammas = np.stack([p_0, p_1])
    Gammas = np.broadcast_to(Gammas, (N,) + Gammas.shape)

    pi_0 = rng.dirichlet(pi_0_a)
    s_0 = rng.choice(pi_0.shape[0], p=pi_0)
    s_tm1 = s_0

    y_samples = np.empty((N,), dtype=np.int64)
    s_samples = np.empty((N,), dtype=np.int64)

    for i in range(N):
        s_t = rng.choice(Gammas.shape[-1], p=Gammas[i, s_tm1])
        s_samples[i] = s_t
        s_tm1 = s_t

        if s_t == 1:
            y_samples[i] = rng.poisson(mu)
        else:
            y_samples[i] = 0

    sample_point = {
        "Y_t": y_samples,
        "p_0": p_0,
        "p_1": p_1,
        "S_t": s_samples,
        "P_tt": Gammas,
        "S_0": s_0,
        "pi_0": pi_0,
    }

    return sample_point


def scipy_logprob(obs, p):
    if p.ndim > 1:
        if p.ndim > obs.ndim:
            obs = obs[((None,) * (p.ndim - obs.ndim) + (Ellipsis,))]
        elif p.ndim < obs.ndim:
            p = p[((None,) * (obs.ndim - p.ndim) + (Ellipsis,))]

        pattern = (p.ndim - 1,) + tuple(range(p.ndim - 1))
        return np.log(np.take_along_axis(p.transpose(pattern), obs, 0))
    else:
        return np.log(p[obs])


def create_aesara_params(dist_params, obs, size):
    dist_params_at = []
    for p in dist_params:
        p_aet = at.as_tensor(p).type()
        p_aet.tag.test_value = p
        dist_params_at.append(p_aet)

    size_at = []
    for s in size:
        s_aet = at.iscalar()
        s_aet.tag.test_value = s
        size_at.append(s_aet)

    obs_at = at.as_tensor(obs).type()
    obs_at.tag.test_value = obs

    return dist_params_at, obs_at, size_at


def scipy_logprob_tester(
    rv_var, obs, dist_params, test_fn=None, check_broadcastable=True, test="logprob"
):
    """Test for correspondence between `RandomVariable` and NumPy shape and
    broadcast dimensions.
    """
    if test_fn is None:
        name = getattr(rv_var.owner.op, "name", None)

        if name is None:
            name = rv_var.__name__

        test_fn = getattr(stats, name)

    if test == "logprob":
        aesara_res = logprob(rv_var, at.as_tensor(obs))
    elif test == "logcdf":
        aesara_res = logcdf(rv_var, at.as_tensor(obs))
    elif test == "icdf":
        aesara_res = icdf(rv_var, at.as_tensor(obs))
    else:
        raise ValueError(f"test must be one of (logprob, logcdf, icdf), got {test}")

    aesara_res_val = aesara_res.eval(dist_params)

    numpy_res = np.asarray(test_fn(obs, *dist_params.values()))

    assert aesara_res.type.numpy_dtype.kind == numpy_res.dtype.kind

    if check_broadcastable:
        numpy_shape = np.shape(numpy_res)
        numpy_bcast = [s == 1 for s in numpy_shape]
        np.testing.assert_array_equal(aesara_res.type.broadcastable, numpy_bcast)

    np.testing.assert_array_equal(aesara_res_val.shape, numpy_res.shape)

    np.testing.assert_array_almost_equal(aesara_res_val, numpy_res, 4)
