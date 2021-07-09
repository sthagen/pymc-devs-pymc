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

import aesara
import aesara.tensor as at
import numpy as np
import pytest

from aesara.tensor.var import TensorConstant

import pymc3 as pm
import pymc3.distributions.transforms as tr

from pymc3.aesaraf import jacobian
from pymc3.distributions import logpt
from pymc3.tests.checks import close_to, close_to_logical
from pymc3.tests.helpers import SeededTest
from pymc3.tests.test_distributions import (
    Circ,
    MultiSimplex,
    R,
    Rminusbig,
    Rplusbig,
    Simplex,
    SortedVector,
    Unit,
    UnitSortedVector,
    Vector,
)

# some transforms (stick breaking) require additon of small slack in order to be numerically
# stable. The minimal addable slack for float32 is higher thus we need to be less strict
tol = 1e-7 if aesara.config.floatX == "float64" else 1e-6


def check_transform(transform, domain, constructor=at.dscalar, test=0, rv_var=None):
    x = constructor("x")
    x.tag.test_value = test
    if rv_var is None:
        rv_var = x
    # test forward and forward_val
    # FIXME: What's being tested here?  That the transformed graph can compile?
    forward_f = aesara.function([x], transform.forward(rv_var, x))
    # test transform identity
    identity_f = aesara.function([x], transform.backward(rv_var, transform.forward(rv_var, x)))
    for val in domain.vals:
        close_to(val, identity_f(val), tol)


def check_vector_transform(transform, domain, rv_var=None):
    return check_transform(transform, domain, at.dvector, test=np.array([0, 0]), rv_var=rv_var)


def get_values(transform, domain=R, constructor=at.dscalar, test=0, rv_var=None):
    x = constructor("x")
    x.tag.test_value = test
    if rv_var is None:
        rv_var = x
    f = aesara.function([x], transform.backward(rv_var, x))
    return np.array([f(val) for val in domain.vals])


def check_jacobian_det(
    transform,
    domain,
    constructor=at.dscalar,
    test=0,
    make_comparable=None,
    elemwise=False,
    rv_var=None,
):
    y = constructor("y")
    y.tag.test_value = test

    if rv_var is None:
        rv_var = y

    x = transform.backward(rv_var, y)
    if make_comparable:
        x = make_comparable(x)

    if not elemwise:
        jac = at.log(at.nlinalg.det(jacobian(x, [y])))
    else:
        jac = at.log(at.abs_(at.diag(jacobian(x, [y]))))

    # ljd = log jacobian det
    actual_ljd = aesara.function([y], jac)

    computed_ljd = aesara.function(
        [y], at.as_tensor_variable(transform.jacobian_det(rv_var, y)), on_unused_input="ignore"
    )

    for yval in domain.vals:
        close_to(actual_ljd(yval), computed_ljd(yval), tol)


def test_stickbreaking():
    check_vector_transform(tr.stick_breaking, Simplex(2))
    check_vector_transform(tr.stick_breaking, Simplex(4))

    check_transform(
        tr.stick_breaking, MultiSimplex(3, 2), constructor=at.dmatrix, test=np.zeros((2, 2))
    )


def test_stickbreaking_bounds():
    vals = get_values(tr.stick_breaking, Vector(R, 2), at.dvector, np.array([0, 0]))

    close_to(vals.sum(axis=1), 1, tol)
    close_to_logical(vals > 0, True, tol)
    close_to_logical(vals < 1, True, tol)

    check_jacobian_det(
        tr.stick_breaking, Vector(R, 2), at.dvector, np.array([0, 0]), lambda x: x[:-1]
    )


def test_stickbreaking_accuracy():
    val = np.array([-30])
    x = at.dvector("x")
    x.tag.test_value = val
    identity_f = aesara.function(
        [x], tr.stick_breaking.forward(x, tr.stick_breaking.backward(x, x))
    )
    close_to(val, identity_f(val), tol)


def test_sum_to_1():
    check_vector_transform(tr.sum_to_1, Simplex(2))
    check_vector_transform(tr.sum_to_1, Simplex(4))

    check_jacobian_det(tr.sum_to_1, Vector(Unit, 2), at.dvector, np.array([0, 0]), lambda x: x[:-1])


def test_log():
    check_transform(tr.log, Rplusbig)

    check_jacobian_det(tr.log, Rplusbig, elemwise=True)
    check_jacobian_det(tr.log, Vector(Rplusbig, 2), at.dvector, [0, 0], elemwise=True)

    vals = get_values(tr.log)
    close_to_logical(vals > 0, True, tol)


def test_log_exp_m1():
    check_transform(tr.log_exp_m1, Rplusbig)

    check_jacobian_det(tr.log_exp_m1, Rplusbig, elemwise=True)
    check_jacobian_det(tr.log_exp_m1, Vector(Rplusbig, 2), at.dvector, [0, 0], elemwise=True)

    vals = get_values(tr.log_exp_m1)
    close_to_logical(vals > 0, True, tol)


def test_logodds():
    check_transform(tr.logodds, Unit)

    check_jacobian_det(tr.logodds, Unit, elemwise=True)
    check_jacobian_det(tr.logodds, Vector(Unit, 2), at.dvector, [0.5, 0.5], elemwise=True)

    vals = get_values(tr.logodds)
    close_to_logical(vals > 0, True, tol)
    close_to_logical(vals < 1, True, tol)


def test_lowerbound():
    def transform_params(rv_var):
        return 0.0, None

    trans = tr.interval(transform_params)
    check_transform(trans, Rplusbig)

    check_jacobian_det(trans, Rplusbig, elemwise=True)
    check_jacobian_det(trans, Vector(Rplusbig, 2), at.dvector, [0, 0], elemwise=True)

    vals = get_values(trans)
    close_to_logical(vals > 0, True, tol)


def test_upperbound():
    def transform_params(rv_var):
        return None, 0.0

    trans = tr.interval(transform_params)
    check_transform(trans, Rminusbig)

    check_jacobian_det(trans, Rminusbig, elemwise=True)
    check_jacobian_det(trans, Vector(Rminusbig, 2), at.dvector, [-1, -1], elemwise=True)

    vals = get_values(trans)
    close_to_logical(vals < 0, True, tol)


def test_interval():
    for a, b in [(-4, 5.5), (0.1, 0.7), (-10, 4.3)]:
        domain = Unit * np.float64(b - a) + np.float64(a)

        def transform_params(x, z=a, y=b):
            return z, y

        trans = tr.interval(transform_params)
        check_transform(trans, domain)

        check_jacobian_det(trans, domain, elemwise=True)

        vals = get_values(trans)
        close_to_logical(vals > a, True, tol)
        close_to_logical(vals < b, True, tol)


@pytest.mark.skipif(aesara.config.floatX == "float32", reason="Test fails on 32 bit")
def test_interval_near_boundary():
    lb = -1.0
    ub = 1e-7
    x0 = np.nextafter(ub, lb)

    with pm.Model() as model:
        pm.Uniform("x", initval=x0, lower=lb, upper=ub)

    log_prob = model.point_logps()
    np.testing.assert_allclose(log_prob, np.array([-52.68]))


def test_circular():
    trans = tr.circular
    check_transform(trans, Circ)

    check_jacobian_det(trans, Circ)

    vals = get_values(trans)
    close_to_logical(vals > -np.pi, True, tol)
    close_to_logical(vals < np.pi, True, tol)

    assert isinstance(trans.forward(None, 1), TensorConstant)


def test_ordered():
    check_vector_transform(tr.ordered, SortedVector(6))

    check_jacobian_det(tr.ordered, Vector(R, 2), at.dvector, np.array([0, 0]), elemwise=False)

    vals = get_values(tr.ordered, Vector(R, 3), at.dvector, np.zeros(3))
    close_to_logical(np.diff(vals) >= 0, True, tol)


def test_chain_values():
    chain_tranf = tr.Chain([tr.logodds, tr.ordered])
    vals = get_values(chain_tranf, Vector(R, 5), at.dvector, np.zeros(5))
    close_to_logical(np.diff(vals) >= 0, True, tol)


def test_chain_vector_transform():
    chain_tranf = tr.Chain([tr.logodds, tr.ordered])
    check_vector_transform(chain_tranf, UnitSortedVector(3))


@pytest.mark.xfail(condition=(aesara.config.floatX == "float32"), reason="Fails on float32")
def test_chain_jacob_det():
    chain_tranf = tr.Chain([tr.logodds, tr.ordered])
    check_jacobian_det(chain_tranf, Vector(R, 4), at.dvector, np.zeros(4), elemwise=False)


class TestElementWiseLogp(SeededTest):
    def build_model(self, distfam, params, size, transform, initval=None):
        if initval is not None:
            initval = pm.floatX(initval)
        with pm.Model() as m:
            distfam("x", size=size, transform=transform, initval=initval, **params)
        return m

    def check_transform_elementwise_logp(self, model):
        x = model.free_RVs[0]
        x0 = x.tag.value_var
        assert x.ndim == logpt(x).ndim

        pt = model.initial_point
        array = np.random.randn(*pt[x0.name].shape)
        transform = x0.tag.transform
        logp_notrans = logpt(x, transform.backward(x, array), transformed=False)

        jacob_det = transform.jacobian_det(x, aesara.shared(array))
        assert logpt(x).ndim == jacob_det.ndim

        v1 = logpt(x, array, jacobian=False).eval()
        v2 = logp_notrans.eval()
        close_to(v1, v2, tol)

    def check_vectortransform_elementwise_logp(self, model, vect_opt=0):
        x = model.free_RVs[0]
        x0 = x.tag.value_var
        assert (x.ndim - 1) == logpt(x).ndim

        pt = model.initial_point
        array = np.random.randn(*pt[x0.name].shape)
        transform = x0.tag.transform
        logp_nojac = logpt(x, transform.backward(x, array), transformed=False)

        jacob_det = transform.jacobian_det(x, aesara.shared(array))
        assert logpt(x).ndim == jacob_det.ndim

        # Hack to get relative tolerance
        a = logpt(x, array.astype(aesara.config.floatX), jacobian=False).eval()
        b = logp_nojac.eval()
        close_to(a, b, np.abs(0.5 * (a + b) * tol))

    @pytest.mark.parametrize(
        "sd,size",
        [
            (2.5, 2),
            (5.0, (2, 3)),
            (np.ones(3) * 10.0, (4, 3)),
        ],
    )
    def test_half_normal(self, sd, size):
        model = self.build_model(pm.HalfNormal, {"sd": sd}, size=size, transform=tr.log)
        self.check_transform_elementwise_logp(model)

    @pytest.mark.parametrize("lam,size", [(2.5, 2), (5.0, (2, 3)), (np.ones(3), (4, 3))])
    def test_exponential(self, lam, size):
        model = self.build_model(pm.Exponential, {"lam": lam}, size=size, transform=tr.log)
        self.check_transform_elementwise_logp(model)

    @pytest.mark.parametrize(
        "a,b,size",
        [
            (1.0, 1.0, 2),
            (0.5, 0.5, (2, 3)),
            (np.ones(3), np.ones(3), (4, 3)),
        ],
    )
    def test_beta(self, a, b, size):
        model = self.build_model(pm.Beta, {"alpha": a, "beta": b}, size=size, transform=tr.logodds)
        self.check_transform_elementwise_logp(model)

    @pytest.mark.parametrize(
        "lower,upper,size",
        [
            (0.0, 1.0, 2),
            (0.5, 5.5, (2, 3)),
            (pm.floatX(np.zeros(3)), pm.floatX(np.ones(3)), (4, 3)),
        ],
    )
    def test_uniform(self, lower, upper, size):
        def transform_params(rv_var):
            _, _, _, lower, upper = rv_var.owner.inputs
            lower = at.as_tensor_variable(lower) if lower is not None else None
            upper = at.as_tensor_variable(upper) if upper is not None else None
            return lower, upper

        interval = tr.Interval(transform_params)
        model = self.build_model(
            pm.Uniform, {"lower": lower, "upper": upper}, size=size, transform=interval
        )
        self.check_transform_elementwise_logp(model)

    @pytest.mark.parametrize(
        "lower, c, upper, size",
        [
            (0.0, 1.0, 2.0, 2),
            (-10, 0, 200, (2, 3)),
            (np.zeros(3), np.ones(3), np.ones(3), (4, 3)),
        ],
    )
    def test_triangular(self, lower, c, upper, size):
        def transform_params(rv_var):
            _, _, _, lower, _, upper = rv_var.owner.inputs
            lower = at.as_tensor_variable(lower) if lower is not None else None
            upper = at.as_tensor_variable(upper) if upper is not None else None
            return lower, upper

        interval = tr.Interval(transform_params)
        model = self.build_model(
            pm.Triangular, {"lower": lower, "c": c, "upper": upper}, size=size, transform=interval
        )
        self.check_transform_elementwise_logp(model)

    @pytest.mark.parametrize(
        "mu,kappa,size", [(0.0, 1.0, 2), (-0.5, 5.5, (2, 3)), (np.zeros(3), np.ones(3), (4, 3))]
    )
    def test_vonmises(self, mu, kappa, size):
        model = self.build_model(
            pm.VonMises, {"mu": mu, "kappa": kappa}, size=size, transform=tr.circular
        )
        self.check_transform_elementwise_logp(model)

    @pytest.mark.parametrize(
        "a,size", [(np.ones(2), None), (np.ones((2, 3)) * 0.5, None), (np.ones(3), (4,))]
    )
    def test_dirichlet(self, a, size):
        model = self.build_model(pm.Dirichlet, {"a": a}, size=size, transform=tr.stick_breaking)
        self.check_vectortransform_elementwise_logp(model, vect_opt=1)

    def test_normal_ordered(self):
        model = self.build_model(
            pm.Normal,
            {"mu": 0.0, "sd": 1.0},
            size=3,
            initval=np.asarray([-1.0, 1.0, 4.0]),
            transform=tr.ordered,
        )
        self.check_vectortransform_elementwise_logp(model, vect_opt=0)

    @pytest.mark.parametrize(
        "sd,size",
        [
            (2.5, (2,)),
            (np.ones(3), (4, 3)),
        ],
    )
    def test_half_normal_ordered(self, sd, size):
        initval = np.sort(np.abs(np.random.randn(*size)))
        model = self.build_model(
            pm.HalfNormal,
            {"sd": sd},
            size=size,
            initval=initval,
            transform=tr.Chain([tr.log, tr.ordered]),
        )
        self.check_vectortransform_elementwise_logp(model, vect_opt=0)

    @pytest.mark.parametrize("lam,size", [(2.5, (2,)), (np.ones(3), (4, 3))])
    def test_exponential_ordered(self, lam, size):
        initval = np.sort(np.abs(np.random.randn(*size)))
        model = self.build_model(
            pm.Exponential,
            {"lam": lam},
            size=size,
            initval=initval,
            transform=tr.Chain([tr.log, tr.ordered]),
        )
        self.check_vectortransform_elementwise_logp(model, vect_opt=0)

    @pytest.mark.parametrize(
        "a,b,size",
        [
            (1.0, 1.0, (2,)),
            (np.ones(3), np.ones(3), (4, 3)),
        ],
    )
    def test_beta_ordered(self, a, b, size):
        initval = np.sort(np.abs(np.random.rand(*size)))
        model = self.build_model(
            pm.Beta,
            {"alpha": a, "beta": b},
            size=size,
            initval=initval,
            transform=tr.Chain([tr.logodds, tr.ordered]),
        )
        self.check_vectortransform_elementwise_logp(model, vect_opt=0)

    @pytest.mark.parametrize(
        "lower,upper,size",
        [(0.0, 1.0, (2,)), (pm.floatX(np.zeros(3)), pm.floatX(np.ones(3)), (4, 3))],
    )
    def test_uniform_ordered(self, lower, upper, size):
        def transform_params(rv_var):
            _, _, _, lower, upper = rv_var.owner.inputs
            lower = at.as_tensor_variable(lower) if lower is not None else None
            upper = at.as_tensor_variable(upper) if upper is not None else None
            return lower, upper

        interval = tr.Interval(transform_params)

        initval = np.sort(np.abs(np.random.rand(*size)))
        model = self.build_model(
            pm.Uniform,
            {"lower": lower, "upper": upper},
            size=size,
            initval=initval,
            transform=tr.Chain([interval, tr.ordered]),
        )
        self.check_vectortransform_elementwise_logp(model, vect_opt=1)

    @pytest.mark.parametrize("mu,kappa,size", [(0.0, 1.0, (2,)), (np.zeros(3), np.ones(3), (4, 3))])
    def test_vonmises_ordered(self, mu, kappa, size):
        initval = np.sort(np.abs(np.random.rand(*size)))
        model = self.build_model(
            pm.VonMises,
            {"mu": mu, "kappa": kappa},
            size=size,
            initval=initval,
            transform=tr.Chain([tr.circular, tr.ordered]),
        )
        self.check_vectortransform_elementwise_logp(model, vect_opt=0)

    @pytest.mark.parametrize(
        "lower,upper,size,transform",
        [
            (0.0, 1.0, (2,), tr.stick_breaking),
            (0.5, 5.5, (2, 3), tr.stick_breaking),
            (np.zeros(3), np.ones(3), (4, 3), tr.Chain([tr.sum_to_1, tr.logodds])),
        ],
    )
    def test_uniform_other(self, lower, upper, size, transform):
        initval = np.ones(size) / size[-1]
        model = self.build_model(
            pm.Uniform,
            {"lower": lower, "upper": upper},
            size=size,
            initval=initval,
            transform=transform,
        )
        self.check_vectortransform_elementwise_logp(model, vect_opt=1)

    @pytest.mark.parametrize(
        "mu,cov,size,shape",
        [
            (np.zeros(2), np.diag(np.ones(2)), None, (2,)),
            (np.zeros(3), np.diag(np.ones(3)), (4,), (4, 3)),
        ],
    )
    def test_mvnormal_ordered(self, mu, cov, size, shape):
        initval = np.sort(np.random.randn(*shape))
        model = self.build_model(
            pm.MvNormal, {"mu": mu, "cov": cov}, size=size, initval=initval, transform=tr.ordered
        )
        self.check_vectortransform_elementwise_logp(model, vect_opt=1)


def test_triangular_transform():
    with pm.Model() as m:
        x = pm.Triangular("x", lower=0, c=1, upper=2)

    transform = x.tag.value_var.tag.transform
    assert np.isclose(transform.backward(x, -np.inf).eval(), 0)
    assert np.isclose(transform.backward(x, np.inf).eval(), 2)
