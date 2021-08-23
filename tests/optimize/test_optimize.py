# Copyright 2018-2020 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Unit tests for the :mod:`pennylane` optimizers.
"""
# pylint: disable=redefined-outer-name
import itertools as it

import numpy as onp
import pytest

import pennylane as qml
from pennylane import numpy as np
from pennylane.optimize import (
    GradientDescentOptimizer,
    MomentumOptimizer,
    NesterovMomentumOptimizer,
    AdagradOptimizer,
    RMSPropOptimizer,
    AdamOptimizer,
    RotoselectOptimizer,
    RotosolveOptimizer,
)


opt_list = [
        AdagradOptimizer,
        AdamOptimizer
        GradientDescentOptimizer,
        MomentumOptimizer,
        NesterovMomentumOptimizer,
        RMSPropOptimzer,
        RotoSelectOptimzier,
        RotosolveOptimizer,
        ShotAdaptiveOptimizer,
    ]

accumulation_vars = []

# Not parameterizing over Rotoselect
@pytest.mark.parametrize(
    "opt",
    [
        AdagradOptimizer(),
        AdamOptimizer()
        GradientDescentOptimizer(),
        MomentumOptimizer(),
        NesterovMomentumOptimizer(),
        RMSPropOptimzer(),
        RotosolveOptimizer(),
        ShotAdaptiveOptimizer(),
    ],
)
class TestOverOpts:
    """Tests keywords, multiple arguments, and non-training arguments in relevent optimizers"""

    def test_kwargs(self, mocker, opt, opt_name, tol):
        """Test that the keywords get passed and alter the function"""

        class func_wrapper:
            @staticmethod
            def func(x, c=1.0):
                return (x - c) ** 2

        x = 1.0

        wrapper = func_wrapper()
        spy = mocker.spy(wrapper, "func")

        x_new_two = opt.step(wrapper.func, x, c=2.0)
        reset(opt)

        args2, kwargs2 = spy.call_args_list[-1]

        x_new_three_wc, cost_three = opt.step_and_cost(wrapper.func, x, c=3.0)
        reset(opt)

        args3, kwargs3 = spy.call_args_list[-1]

        if isinstance(opt, RotosolveOptimizer):
            assert args2 == (x,)
            assert args3 == (x,)
        else:
            assert x_new_two != pytest.approx(x, abs=tol)
            assert x_new_three_wc != pytest.approx(x, abs=tol)

        assert kwargs2 == {"c": 2.0}
        assert kwargs3 == {"c": 3.0}

        assert cost_three == pytest.approx(wrapper.func(x, c=3.0), abs=tol)

    def test_multi_args(self, mocker, opt, opt_name, tol):
        """Test passing multiple arguments to function"""

        class func_wrapper:
            @staticmethod
            def func(x, y, z):
                return x[0] * y[0] + z[0]

        wrapper = func_wrapper()
        spy = mocker.spy(wrapper, "func")

        x = np.array([1.0])
        y = np.array([2.0])
        z = np.array([3.0])

        (x_new, y_new, z_new), cost = opt.step_and_cost(wrapper.func, x, y, z)
        reset(opt)
        args_called1, kwargs1 = spy.call_args_list[-1]  # just take last call

        x_new2, y_new2, z_new2 = opt.step(wrapper.func, x_new, y_new, z_new)
        reset(opt)
        args_called2, kwargs2 = spy.call_args_list[-1]  # just take last call

        if opt_name != "roto":
            assert args_called1 == (x, y, z)
            assert args_called2 == (x_new, y_new, z_new)
        else:
            assert not np.allclose(x_new, x, tol=tol)
            assert not np.allclose(y_new, y, tol=tol)
            assert not np.allclose(z_new, z, tol=tol)

        assert kwargs1 == {}
        assert kwargs2 == {}

        assert cost == pytest.approx(wrapper.func(x, y, z), abs=tol)

    def test_nontrainable_data(self, opt, opt_name, tol):
        """Check non-trainable argument does not get updated"""

        def func(x, data):
            return x[0] * data[0]

        x = np.array([1.0])
        data = np.array([1.0], requires_grad=False)

        args_new = opt.step(func, x, data)
        reset(opt)
        args_new_wc, cost = opt.step_and_cost(func, *args_new)
        reset(opt)

        assert len(args_new) == pytest.approx(2, abs=tol)
        assert args_new[0] != pytest.approx(x, abs=tol)
        assert args_new[1] == pytest.approx(data, abs=tol)

        assert cost == pytest.approx(func(*args_new), abs=tol)

    def test_steps_the_same(self, opt, opt_name, tol):
        """Tests whether separating the args into different inputs affects their
        optimization step. Assumes single argument optimization is correct, as tested elsewhere."""

        def func1(x, y, z):
            return x[0] * y[0] * z[0]

        def func2(args):
            return args[0][0] * args[1][0] * args[2][0]

        x = np.array([1.0])
        y = np.array([2.0])
        z = np.array([3.0])
        args = (x, y, z)

        x_seperate, y_seperate, z_seperate = opt.step(func1, x, y, z)
        reset(opt)

        args_new = opt.step(func2, args)
        reset(opt)

        assert x_seperate == pytest.approx(args_new[0], abs=tol)
        assert y_seperate == pytest.approx(args_new[1], abs=tol)
        assert z_seperate == pytest.approx(args_new[2], abs=tol)

    def test_one_trainable_one_non_trainable(self, opt, opt_name, tol):
        """Tests that a cost function that takes one trainable and one
        non-trainable parameter executes well."""
        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev)
        def circuit(x):
            qml.RX(x, wires=0)
            return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

        def cost(x, target):
            return (circuit(x) - target) ** 2

        ev = np.tensor(0.7781, requires_grad=False)
        x = np.tensor(0.0, requires_grad=True)

        original_ev = ev

        (x, ev), cost = opt.step_and_cost(cost, x, ev)

        # check that the argument to RX doesn't change, as the X rotation doesn't influence <Z>
        assert x == 0
        assert ev == original_ev

    def test_one_non_trainable_one_trainable(self, opt, opt_name, tol):
        """Tests that a cost function that takes one non-trainable and one
        trainable parameter executes well."""
        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev)
        def circuit(x):
            qml.RX(x, wires=0)
            return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

        def cost(target, x):  # Note: the order of the arguments has been swapped
            return (circuit(x) - target) ** 2

        ev = np.tensor(0.7781, requires_grad=False)
        x = np.tensor(0.0, requires_grad=True)

        original_ev = ev

        (ev, x), cost = opt.step_and_cost(cost, ev, x)
        # check that the argument to RX doesn't change, as the X rotation doesn't influence <Z>
        assert x == 0
        assert ev == original_ev

    def test_two_trainable_args(self, opt, opt_name, tol):
        """Tests that a cost function that takes at least two trainable
        arguments executes well."""
        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev)
        def circuit(x, y):
            qml.RX(x, wires=0)
            qml.RX(y, wires=0)
            return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

        def cost(x, y, target):
            return (circuit(x, y) - target) ** 2

        ev = np.tensor(0.7781, requires_grad=False)
        x = np.tensor(0.0, requires_grad=True)
        y = np.tensor(0.0, requires_grad=True)

        original_ev = ev

        (x, y, ev), cost = opt.step_and_cost(cost, x, y, ev)

        # check that the argument to RX doesn't change, as the X rotation doesn't influence <Z>
        assert x == 0
        assert ev == original_ev
