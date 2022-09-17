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

import io
import itertools as it

import aesara
import aesara.tensor as at
import cloudpickle
import numpy as np
import pytest
import scipy.stats as st

from aesara import shared
from aesara.tensor.var import TensorVariable

import pymc as pm

from pymc.aesaraf import GeneratorOp, floatX
from pymc.tests.helpers import SeededTest, select_by_precision


class TestData(SeededTest):
    def test_deterministic(self):
        data_values = np.array([0.5, 0.4, 5, 2])
        with pm.Model() as model:
            X = pm.MutableData("X", data_values)
            pm.Normal("y", 0, 1, observed=X)
            model.compile_logp()(model.initial_point())

    def test_sample(self):
        x = np.random.normal(size=100)
        y = x + np.random.normal(scale=1e-2, size=100)

        x_pred = np.linspace(-3, 3, 200, dtype="float32")

        with pm.Model():
            x_shared = pm.MutableData("x_shared", x)
            b = pm.Normal("b", 0.0, 10.0)
            pm.Normal("obs", b * x_shared, np.sqrt(1e-2), observed=y, shape=x_shared.shape)

            prior_trace0 = pm.sample_prior_predictive(1000)
            idata = pm.sample(1000, tune=1000, chains=1)
            pp_trace0 = pm.sample_posterior_predictive(idata)

            x_shared.set_value(x_pred)
            prior_trace1 = pm.sample_prior_predictive(1000)
            pp_trace1 = pm.sample_posterior_predictive(idata)

        assert prior_trace0.prior["b"].shape == (1, 1000)
        assert prior_trace0.prior_predictive["obs"].shape == (1, 1000, 100)
        assert prior_trace1.prior_predictive["obs"].shape == (1, 1000, 200)

        assert pp_trace0.posterior_predictive["obs"].shape == (1, 1000, 100)
        np.testing.assert_allclose(
            x, pp_trace0.posterior_predictive["obs"].mean(("chain", "draw")), atol=1e-1
        )

        assert pp_trace1.posterior_predictive["obs"].shape == (1, 1000, 200)
        np.testing.assert_allclose(
            x_pred, pp_trace1.posterior_predictive["obs"].mean(("chain", "draw")), atol=1e-1
        )

    def test_sample_posterior_predictive_after_set_data(self):
        with pm.Model() as model:
            x = pm.MutableData("x", [1.0, 2.0, 3.0])
            y = pm.ConstantData("y", [1.0, 2.0, 3.0])
            beta = pm.Normal("beta", 0, 10.0)
            pm.Normal("obs", beta * x, np.sqrt(1e-2), observed=y)
            trace = pm.sample(
                1000,
                tune=1000,
                chains=1,
                return_inferencedata=False,
                compute_convergence_checks=False,
            )
        # Predict on new data.
        with model:
            x_test = [5, 6, 9]
            pm.set_data(new_data={"x": x_test})
            y_test = pm.sample_posterior_predictive(trace)

        assert y_test.posterior_predictive["obs"].shape == (1, 1000, 3)
        np.testing.assert_allclose(
            x_test, y_test.posterior_predictive["obs"].mean(("chain", "draw")), atol=1e-1
        )

    def test_sample_posterior_predictive_after_set_data_with_coords(self):
        y = np.array([1.0, 2.0, 3.0])
        with pm.Model() as model:
            x = pm.MutableData("x", [1.0, 2.0, 3.0], dims="obs_id")
            beta = pm.Normal("beta", 0, 10.0)
            pm.Normal("obs", beta * x, np.sqrt(1e-3), observed=y, dims="obs_id")
            idata = pm.sample(
                10,
                tune=100,
                chains=1,
                return_inferencedata=True,
                compute_convergence_checks=False,
            )
        # Predict on new data.
        with model:
            x_test = [5, 6]
            pm.set_data(new_data={"x": x_test}, coords={"obs_id": ["a", "b"]})
            pm.sample_posterior_predictive(idata, extend_inferencedata=True, predictions=True)

        assert idata.predictions["obs"].shape == (1, 10, 2)
        assert np.all(idata.predictions["obs_id"].values == np.array(["a", "b"]))
        np.testing.assert_allclose(
            x_test, idata.predictions["obs"].mean(("chain", "draw")), atol=1e-1
        )

    def test_sample_after_set_data(self):
        with pm.Model() as model:
            x = pm.MutableData("x", [1.0, 2.0, 3.0])
            y = pm.MutableData("y", [1.0, 2.0, 3.0])
            beta = pm.Normal("beta", 0, 10.0)
            pm.Normal("obs", beta * x, np.sqrt(1e-2), observed=y)
            pm.sample(
                1000,
                tune=1000,
                chains=1,
                compute_convergence_checks=False,
            )
        # Predict on new data.
        new_x = [5.0, 6.0, 9.0]
        new_y = [5.0, 6.0, 9.0]
        with model:
            pm.set_data(new_data={"x": new_x, "y": new_y})
            new_idata = pm.sample(
                1000,
                tune=1000,
                chains=1,
                compute_convergence_checks=False,
            )
            pp_trace = pm.sample_posterior_predictive(new_idata)

        assert pp_trace.posterior_predictive["obs"].shape == (1, 1000, 3)
        np.testing.assert_allclose(
            new_y, pp_trace.posterior_predictive["obs"].mean(("chain", "draw")), atol=1e-1
        )

    def test_shared_data_as_index(self):
        """
        Allow pm.Data to be used for index variables, i.e with integers as well as floats.
        See https://github.com/pymc-devs/pymc/issues/3813
        """
        with pm.Model() as model:
            index = pm.MutableData("index", [2, 0, 1, 0, 2])
            y = pm.MutableData("y", [1.0, 2.0, 3.0, 2.0, 1.0])
            alpha = pm.Normal("alpha", 0, 1.5, size=3)
            pm.Normal("obs", alpha[index], np.sqrt(1e-2), observed=y)

            prior_trace = pm.sample_prior_predictive(1000)
            idata = pm.sample(
                1000,
                tune=1000,
                chains=1,
                compute_convergence_checks=False,
            )

        # Predict on new data
        new_index = np.array([0, 1, 2])
        new_y = [5.0, 6.0, 9.0]
        with model:
            pm.set_data(new_data={"index": new_index, "y": new_y})
            pp_trace = pm.sample_posterior_predictive(idata, var_names=["alpha", "obs"])

        assert prior_trace.prior["alpha"].shape == (1, 1000, 3)
        assert idata.posterior["alpha"].shape == (1, 1000, 3)
        assert pp_trace.posterior_predictive["alpha"].shape == (1, 1000, 3)
        assert pp_trace.posterior_predictive["obs"].shape == (1, 1000, 3)

    def test_shared_data_as_rv_input(self):
        """
        Allow pm.Data to be used as input for other RVs.
        See https://github.com/pymc-devs/pymc/issues/3842
        """
        with pm.Model() as m:
            x = pm.MutableData("x", [1.0, 2.0, 3.0])
            y = pm.Normal("y", mu=x, size=(2, 3))
            assert y.eval().shape == (2, 3)
            idata = pm.sample(
                chains=1,
                tune=500,
                draws=550,
                return_inferencedata=True,
                compute_convergence_checks=False,
            )
        samples = idata.posterior["y"]
        assert samples.shape == (1, 550, 2, 3)

        np.testing.assert_allclose(np.array([1.0, 2.0, 3.0]), x.get_value(), atol=1e-1)
        np.testing.assert_allclose(
            np.array([1.0, 2.0, 3.0]), samples.mean(("chain", "draw", "y_dim_0")), atol=1e-1
        )

        with m:
            pm.set_data({"x": np.array([2.0, 4.0, 6.0])})
            assert y.eval().shape == (2, 3)
            idata = pm.sample(
                chains=1,
                tune=500,
                draws=620,
                return_inferencedata=True,
                compute_convergence_checks=False,
            )
        samples = idata.posterior["y"]
        assert samples.shape == (1, 620, 2, 3)

        np.testing.assert_allclose(np.array([2.0, 4.0, 6.0]), x.get_value(), atol=1e-1)
        np.testing.assert_allclose(
            np.array([2.0, 4.0, 6.0]), samples.mean(("chain", "draw", "y_dim_0")), atol=1e-1
        )

    def test_shared_scalar_as_rv_input(self):
        # See https://github.com/pymc-devs/pymc/issues/3139
        with pm.Model() as m:
            shared_var = shared(5.0)
            v = pm.Normal("v", mu=shared_var, size=1)

        m_logp_fn = m.compile_logp()

        np.testing.assert_allclose(
            m_logp_fn({"v": np.r_[5.0]}),
            -0.91893853,
            rtol=1e-5,
        )

        shared_var.set_value(10.0)

        np.testing.assert_allclose(
            m_logp_fn({"v": np.r_[10.0]}),
            -0.91893853,
            rtol=1e-5,
        )

    def test_creation_of_data_outside_model_context(self):
        with pytest.raises((IndexError, TypeError)) as error:
            pm.ConstantData("data", [1.1, 2.2, 3.3])
        error.match("No model on context stack")

    def test_set_data_to_non_data_container_variables(self):
        with pm.Model() as model:
            x = np.array([1.0, 2.0, 3.0])
            y = np.array([1.0, 2.0, 3.0])
            beta = pm.Normal("beta", 0, 10.0)
            pm.Normal("obs", beta * x, np.sqrt(1e-2), observed=y)
            pm.sample(
                1000,
                tune=1000,
                chains=1,
                compute_convergence_checks=False,
            )
        with pytest.raises(TypeError) as error:
            pm.set_data({"beta": [1.1, 2.2, 3.3]}, model=model)
        error.match("The variable `beta` must be a `SharedVariable`")

    @pytest.mark.xfail(reason="Depends on ModelGraph")
    def test_model_to_graphviz_for_model_with_data_container(self):
        with pm.Model() as model:
            x = pm.ConstantData("x", [1.0, 2.0, 3.0])
            y = pm.MutableData("y", [1.0, 2.0, 3.0])
            beta = pm.Normal("beta", 0, 10.0)
            obs_sigma = floatX(np.sqrt(1e-2))
            pm.Normal("obs", beta * x, obs_sigma, observed=y)
            pm.sample(
                1000,
                tune=1000,
                chains=1,
                compute_convergence_checks=False,
            )

        for formatting in {"latex", "latex_with_params"}:
            with pytest.raises(ValueError, match="Unsupported formatting"):
                pm.model_to_graphviz(model, formatting=formatting)

        exp_without = [
            'x [label="x\n~\nConstantData" shape=box style="rounded, filled"]',
            'y [label="x\n~\nMutableData" shape=box style="rounded, filled"]',
            'beta [label="beta\n~\nNormal"]',
            'obs [label="obs\n~\nNormal" style=filled]',
        ]
        exp_with = [
            'x [label="x\n~\nConstantData" shape=box style="rounded, filled"]',
            'y [label="x\n~\nMutableData" shape=box style="rounded, filled"]',
            'beta [label="beta\n~\nNormal(mu=0.0, sigma=10.0)"]',
            f'obs [label="obs\n~\nNormal(mu=f(f(beta), x), sigma={obs_sigma})" style=filled]',
        ]
        for formatting, expected_substrings in [
            ("plain", exp_without),
            ("plain_with_params", exp_with),
        ]:
            g = pm.model_to_graphviz(model, formatting=formatting)
            # check formatting of RV nodes
            for expected in expected_substrings:
                assert expected in g.source

    def test_explicit_coords(self):
        N_rows = 5
        N_cols = 7
        data = np.random.uniform(size=(N_rows, N_cols))
        coords = {
            "rows": [f"R{r+1}" for r in range(N_rows)],
            "columns": [f"C{c+1}" for c in range(N_cols)],
        }
        # pass coordinates explicitly, use numpy array in Data container
        with pm.Model(coords=coords) as pmodel:
            # Dims created from coords are constant by default
            assert isinstance(pmodel.dim_lengths["rows"], at.TensorConstant)
            assert isinstance(pmodel.dim_lengths["columns"], at.TensorConstant)
            pm.MutableData("observations", data, dims=("rows", "columns"))
            # new data with same (!) shape
            pm.set_data({"observations": data + 1})
            # new data with same (!) shape and coords
            pm.set_data({"observations": data}, coords=coords)
        assert "rows" in pmodel.coords
        assert pmodel.coords["rows"] == ("R1", "R2", "R3", "R4", "R5")
        assert "rows" in pmodel.dim_lengths
        assert pmodel.dim_lengths["rows"].eval() == 5
        assert "columns" in pmodel.coords
        assert pmodel.coords["columns"] == ("C1", "C2", "C3", "C4", "C5", "C6", "C7")
        assert pmodel.RV_dims == {"observations": ("rows", "columns")}
        assert "columns" in pmodel.dim_lengths
        assert pmodel.dim_lengths["columns"].eval() == 7

    def test_set_coords_through_pmdata(self):
        with pm.Model() as pmodel:
            pm.ConstantData(
                "population", [100, 200], dims="city", coords={"city": ["Tinyvil", "Minitown"]}
            )
            pm.MutableData(
                "temperature",
                [[15, 20, 22, 17], [18, 22, 21, 12]],
                dims=("city", "season"),
                coords={"season": ["winter", "spring", "summer", "fall"]},
            )
        assert "city" in pmodel.coords
        assert "season" in pmodel.coords
        assert pmodel.coords["city"] == ("Tinyvil", "Minitown")
        assert pmodel.coords["season"] == ("winter", "spring", "summer", "fall")

    def test_symbolic_coords(self):
        """
        In v4 dimensions can be created without passing coordinate values.
        Their lengths are then automatically linked to the corresponding Tensor dimension.
        """
        with pm.Model() as pmodel:
            # Dims created from MutableData are TensorVariables linked to the SharedVariable.shape
            intensity = pm.MutableData("intensity", np.ones((2, 3)), dims=("row", "column"))
            assert "row" in pmodel.dim_lengths
            assert "column" in pmodel.dim_lengths
            assert isinstance(pmodel.dim_lengths["row"], TensorVariable)
            assert isinstance(pmodel.dim_lengths["column"], TensorVariable)
            assert pmodel.dim_lengths["row"].eval() == 2
            assert pmodel.dim_lengths["column"].eval() == 3

            intensity.set_value(floatX(np.ones((4, 5))))
            assert pmodel.dim_lengths["row"].eval() == 4
            assert pmodel.dim_lengths["column"].eval() == 5

    def test_implicit_coords_series(self):
        pd = pytest.importorskip("pandas")
        ser_sales = pd.Series(
            data=np.random.randint(low=0, high=30, size=22),
            index=pd.date_range(start="2020-05-01", periods=22, freq="24H", name="date"),
            name="sales",
        )
        with pm.Model() as pmodel:
            pm.ConstantData("sales", ser_sales, dims="date", export_index_as_coords=True)

        assert "date" in pmodel.coords
        assert len(pmodel.coords["date"]) == 22
        assert pmodel.RV_dims == {"sales": ("date",)}

    def test_implicit_coords_dataframe(self):
        pd = pytest.importorskip("pandas")
        N_rows = 5
        N_cols = 7
        df_data = pd.DataFrame()
        for c in range(N_cols):
            df_data[f"Column {c+1}"] = np.random.normal(size=(N_rows,))
        df_data.index.name = "rows"
        df_data.columns.name = "columns"

        # infer coordinates from index and columns of the DataFrame
        with pm.Model() as pmodel:
            pm.ConstantData(
                "observations", df_data, dims=("rows", "columns"), export_index_as_coords=True
            )

        assert "rows" in pmodel.coords
        assert "columns" in pmodel.coords
        assert pmodel.RV_dims == {"observations": ("rows", "columns")}

    def test_data_kwargs(self):
        strict_value = True
        allow_downcast_value = False
        with pm.Model():
            data = pm.MutableData(
                "mdata",
                value=[[1.0], [2.0], [3.0]],
                strict=strict_value,
                allow_downcast=allow_downcast_value,
            )
        assert data.container.strict is strict_value
        assert data.container.allow_downcast is allow_downcast_value

    def test_data_mutable_default_warning(self):
        with pm.Model():
            with pytest.warns(UserWarning, match="`mutable` kwarg was not specified"):
                data = pm.Data("x", [1, 2, 3])
            assert isinstance(data, at.TensorConstant)
        pass


def test_data_naming():
    """
    This is a test for issue #3793 -- `Data` objects in named models are
    not given model-relative names.
    """
    with pm.Model("named_model") as model:
        x = pm.ConstantData("x", [1.0, 2.0, 3.0])
        y = pm.Normal("y")
    assert y.name == "named_model::y"
    assert x.name == "named_model::x"


def test_get_data():
    data = pm.get_data("radon.csv")
    assert type(data) == io.BytesIO


class _DataSampler:
    """
    Not for users
    """

    def __init__(self, data, batchsize=50, random_seed=42, dtype="floatX"):
        self.dtype = aesara.config.floatX if dtype == "floatX" else dtype
        self.rng = np.random.RandomState(random_seed)
        self.data = data
        self.n = batchsize

    def __iter__(self):
        return self

    def __next__(self):
        idx = self.rng.uniform(size=self.n, low=0.0, high=self.data.shape[0] - 1e-16).astype(
            "int64"
        )
        return np.asarray(self.data[idx], self.dtype)

    next = __next__


@pytest.fixture(scope="module")
def datagen():
    return _DataSampler(np.random.uniform(size=(1000, 10)))


def integers():
    i = 0
    while True:
        yield pm.floatX(i)
        i += 1


def integers_ndim(ndim):
    i = 0
    while True:
        yield np.ones((2,) * ndim) * i
        i += 1


@pytest.mark.usefixtures("strict_float32")
class TestGenerator:
    def test_basic(self):
        generator = pm.GeneratorAdapter(integers())
        gop = GeneratorOp(generator)()
        assert gop.tag.test_value == np.float32(0)
        f = aesara.function([], gop)
        assert f() == np.float32(0)
        assert f() == np.float32(1)
        for _ in range(2, 100):
            f()
        assert f() == np.float32(100)

    def test_ndim(self):
        for ndim in range(10):
            res = list(it.islice(integers_ndim(ndim), 0, 2))
            generator = pm.GeneratorAdapter(integers_ndim(ndim))
            gop = GeneratorOp(generator)()
            f = aesara.function([], gop)
            assert ndim == res[0].ndim
            np.testing.assert_equal(f(), res[0])
            np.testing.assert_equal(f(), res[1])

    def test_cloning_available(self):
        gop = pm.generator(integers())
        res = gop**2
        shared = aesara.shared(pm.floatX(10))
        res1 = aesara.clone_replace(res, {gop: shared})
        f = aesara.function([], res1)
        assert f() == np.float32(100)

    def test_default_value(self):
        def gen():
            for i in range(2):
                yield pm.floatX(np.ones((10, 10)) * i)

        gop = pm.generator(gen(), np.ones((10, 10)) * 10)
        f = aesara.function([], gop)
        np.testing.assert_equal(np.ones((10, 10)) * 0, f())
        np.testing.assert_equal(np.ones((10, 10)) * 1, f())
        np.testing.assert_equal(np.ones((10, 10)) * 10, f())
        with pytest.raises(ValueError):
            gop.set_default(1)

    def test_set_gen_and_exc(self):
        def gen():
            for i in range(2):
                yield pm.floatX(np.ones((10, 10)) * i)

        gop = pm.generator(gen())
        f = aesara.function([], gop)
        np.testing.assert_equal(np.ones((10, 10)) * 0, f())
        np.testing.assert_equal(np.ones((10, 10)) * 1, f())
        with pytest.raises(StopIteration):
            f()
        gop.set_gen(gen())
        np.testing.assert_equal(np.ones((10, 10)) * 0, f())
        np.testing.assert_equal(np.ones((10, 10)) * 1, f())

    def test_pickling(self, datagen):
        gen = pm.generator(datagen)
        cloudpickle.loads(cloudpickle.dumps(gen))
        bad_gen = pm.generator(integers())
        with pytest.raises(TypeError):
            cloudpickle.dumps(bad_gen)

    def test_gen_cloning_with_shape_change(self, datagen):
        gen = pm.generator(datagen)
        gen_r = pm.at_rng().normal(size=gen.shape).T
        X = gen.dot(gen_r)
        res, _ = aesara.scan(lambda x: x.sum(), X, n_steps=X.shape[0])
        assert res.eval().shape == (50,)
        shared = aesara.shared(datagen.data.astype(gen.dtype))
        res2 = aesara.clone_replace(res, {gen: shared**2})
        assert res2.eval().shape == (1000,)


def gen1():
    i = 0
    while True:
        yield np.ones((10, 100)) * i
        i += 1


def gen2():
    i = 0
    while True:
        yield np.ones((20, 100)) * i
        i += 1


class TestScaling:
    """
    Related to minibatch training
    """

    def test_density_scaling(self):
        with pm.Model() as model1:
            pm.Normal("n", observed=[[1]], total_size=1)
            p1 = aesara.function([], model1.logp())

        with pm.Model() as model2:
            pm.Normal("n", observed=[[1]], total_size=2)
            p2 = aesara.function([], model2.logp())
        assert p1() * 2 == p2()

    def test_density_scaling_with_generator(self):
        # We have different size generators

        def true_dens():
            g = gen1()
            for i, point in enumerate(g):
                yield st.norm.logpdf(point).sum() * 10

        t = true_dens()
        # We have same size models
        with pm.Model() as model1:
            pm.Normal("n", observed=gen1(), total_size=100)
            p1 = aesara.function([], model1.logp())

        with pm.Model() as model2:
            gen_var = pm.generator(gen2())
            pm.Normal("n", observed=gen_var, total_size=100)
            p2 = aesara.function([], model2.logp())

        for i in range(10):
            _1, _2, _t = p1(), p2(), next(t)
            decimals = select_by_precision(float64=7, float32=1)
            np.testing.assert_almost_equal(_1, _t, decimal=decimals)  # Value O(-50,000)
            np.testing.assert_almost_equal(_1, _2)
        # Done

    def test_gradient_with_scaling(self):
        with pm.Model() as model1:
            genvar = pm.generator(gen1())
            m = pm.Normal("m")
            pm.Normal("n", observed=genvar, total_size=1000)
            grad1 = aesara.function([m.tag.value_var], at.grad(model1.logp(), m.tag.value_var))
        with pm.Model() as model2:
            m = pm.Normal("m")
            shavar = aesara.shared(np.ones((1000, 100)))
            pm.Normal("n", observed=shavar)
            grad2 = aesara.function([m.tag.value_var], at.grad(model2.logp(), m.tag.value_var))

        for i in range(10):
            shavar.set_value(np.ones((100, 100)) * i)
            g1 = grad1(1)
            g2 = grad2(1)
            np.testing.assert_almost_equal(g1, g2)

    def test_multidim_scaling(self):
        with pm.Model() as model0:
            pm.Normal("n", observed=[[1, 1], [1, 1]], total_size=[])
            p0 = aesara.function([], model0.logp())

        with pm.Model() as model1:
            pm.Normal("n", observed=[[1, 1], [1, 1]], total_size=[2, 2])
            p1 = aesara.function([], model1.logp())

        with pm.Model() as model2:
            pm.Normal("n", observed=[[1], [1]], total_size=[2, 2])
            p2 = aesara.function([], model2.logp())

        with pm.Model() as model3:
            pm.Normal("n", observed=[[1, 1]], total_size=[2, 2])
            p3 = aesara.function([], model3.logp())

        with pm.Model() as model4:
            pm.Normal("n", observed=[[1]], total_size=[2, 2])
            p4 = aesara.function([], model4.logp())

        with pm.Model() as model5:
            pm.Normal("n", observed=[[1]], total_size=[2, Ellipsis, 2])
            p5 = aesara.function([], model5.logp())
        _p0 = p0()
        assert (
            np.allclose(_p0, p1())
            and np.allclose(_p0, p2())
            and np.allclose(_p0, p3())
            and np.allclose(_p0, p4())
            and np.allclose(_p0, p5())
        )

    def test_common_errors(self):
        with pytest.raises(ValueError) as e:
            with pm.Model() as m:
                pm.Normal("n", observed=[[1]], total_size=[2, Ellipsis, 2, 2])
                m.logp()
        assert "Length of" in str(e.value)
        with pytest.raises(ValueError) as e:
            with pm.Model() as m:
                pm.Normal("n", observed=[[1]], total_size=[2, 2, 2])
                m.logp()
        assert "Length of" in str(e.value)
        with pytest.raises(TypeError) as e:
            with pm.Model() as m:
                pm.Normal("n", observed=[[1]], total_size="foo")
                m.logp()
        assert "Unrecognized" in str(e.value)
        with pytest.raises(TypeError) as e:
            with pm.Model() as m:
                pm.Normal("n", observed=[[1]], total_size=["foo"])
                m.logp()
        assert "Unrecognized" in str(e.value)
        with pytest.raises(ValueError) as e:
            with pm.Model() as m:
                pm.Normal("n", observed=[[1]], total_size=[Ellipsis, Ellipsis])
                m.logp()
        assert "Double Ellipsis" in str(e.value)

    def test_mixed1(self):
        with pm.Model():
            data = np.random.rand(10, 20, 30, 40, 50)
            mb = pm.Minibatch(data, [2, None, 20, Ellipsis, 10])
            pm.Normal("n", observed=mb, total_size=(10, None, 30, Ellipsis, 50))

    def test_mixed2(self):
        with pm.Model():
            data = np.random.rand(10, 20, 30, 40, 50)
            mb = pm.Minibatch(data, [2, None, 20])
            pm.Normal("n", observed=mb, total_size=(10, None, 30))

    def test_free_rv(self):
        with pm.Model() as model4:
            pm.Normal("n", observed=[[1, 1], [1, 1]], total_size=[2, 2])
            p4 = aesara.function([], model4.logp())

        with pm.Model() as model5:
            n = pm.Normal("n", total_size=[2, Ellipsis, 2], size=(2, 2))
            p5 = aesara.function([n.tag.value_var], model5.logp())
        assert p4() == p5(pm.floatX([[1]]))
        assert p4() == p5(pm.floatX([[1, 1], [1, 1]]))


@pytest.mark.usefixtures("strict_float32")
class TestMinibatch:
    data = np.random.rand(30, 10, 40, 10, 50)

    def test_1d(self):
        mb = pm.Minibatch(self.data, 20)
        assert mb.eval().shape == (20, 10, 40, 10, 50)

    def test_2d(self):
        mb = pm.Minibatch(self.data, [(10, 42), (4, 42)])
        assert mb.eval().shape == (10, 4, 40, 10, 50)

    @pytest.mark.parametrize(
        "batch_size, expected",
        [
            ([(10, 42), None, (4, 42)], (10, 10, 4, 10, 50)),
            ([(10, 42), Ellipsis, (4, 42)], (10, 10, 40, 10, 4)),
            ([(10, 42), None, Ellipsis, (4, 42)], (10, 10, 40, 10, 4)),
            ([10, None, Ellipsis, (4, 42)], (10, 10, 40, 10, 4)),
        ],
    )
    def test_special_batch_size(self, batch_size, expected):
        mb = pm.Minibatch(self.data, batch_size)
        assert mb.eval().shape == expected

    def test_cloning_available(self):
        gop = pm.Minibatch(np.arange(100), 1)
        res = gop**2
        shared = aesara.shared(np.array([10]))
        res1 = aesara.clone_replace(res, {gop: shared})
        f = aesara.function([], res1)
        assert f() == np.array([100])

    def test_align(self):
        m = pm.Minibatch(np.arange(1000), 1, random_seed=1)
        n = pm.Minibatch(np.arange(1000), 1, random_seed=1)
        f = aesara.function([], [m, n])
        n.eval()  # not aligned
        a, b = zip(*(f() for _ in range(1000)))
        assert a != b
        pm.align_minibatches()
        a, b = zip(*(f() for _ in range(1000)))
        assert a == b
        n.eval()  # not aligned
        pm.align_minibatches([m])
        a, b = zip(*(f() for _ in range(1000)))
        assert a != b
        pm.align_minibatches([m, n])
        a, b = zip(*(f() for _ in range(1000)))
        assert a == b
