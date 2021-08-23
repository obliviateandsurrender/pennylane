# Copyright 2018-2021 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests specific to the MomentumOptimizer and NesterovMomentumOptimizer"""

import pytest
import numpy as np

import pennylane as qml

from pennylane.optimize import MomentumOptimizer, NesterovMomentumOptimizer

@pytest.mark.parameterize("opt_class", (MomentumOptimizer, NesterovMomentumOptimizer))
class TestBothMomentumTypes:

    def test_stepsize_momentum_control(self, opt_class):
        
        opt = opt_class(stepsize=1.0, momentum=1.0)

        assert opt._stepsize == 1.0
        assert opt.momentum == 1.0

        opt.update_stepsize(stepsize=2.0)

        assert opt._stepsize == 2.0

    def test_reset_accumulation(self, opt_class):

        opt = opt_class()

        opt.accumulation = "some value"

        opt.reset()

        assert opt.accumulation is None