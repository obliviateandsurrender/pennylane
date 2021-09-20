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
"""
This submodule contains the discrete-variable quantum operations that come
from quantum chemistry applications.
"""
# pylint:disable=abstract-method,arguments-differ,protected-access
import cmath
import math
import numpy as np

import pennylane as qml
from pennylane.operation import Operation

INV_SQRT2 = 1 / math.sqrt(2)

# Four term gradient recipe for controlled rotations
c1 = INV_SQRT2 * (np.sqrt(2) + 1) / 4
c2 = INV_SQRT2 * (np.sqrt(2) - 1) / 4
a = np.pi / 2
b = 3 * np.pi / 2
four_term_grad_recipe = ([[c1, 1, a], [-c1, 1, -a], [-c2, 1, b], [c2, 1, -b]],)


class SingleExcitation(Operation):
    r"""SingleExcitation(phi, wires)
    Single excitation rotation.

    .. math:: U(\phi) = \begin{bmatrix}
                1 & 0 & 0 & 0 \\
                0 & \cos(\phi/2) & -\sin(\phi/2) & 0 \\
                0 & \sin(\phi/2) & \cos(\phi/2) & 0 \\
                0 & 0 & 0 & 1
            \end{bmatrix}.

    This operation performs a rotation in the two-dimensional subspace :math:`\{|01\rangle,
    |10\rangle\}`. The name originates from the occupation-number representation of
    fermionic wavefunctions, where the transformation  from :math:`|10\rangle` to :math:`|01\rangle`
    is interpreted as "exciting" a particle from the first qubit to the second.

    **Details:**

    * Number of wires: 2
    * Number of parameters: 1
    * Gradient recipe: The ``SingleExcitation`` operator satisfies a four-term parameter-shift rule
      (see Appendix F, https://arxiv.org/abs/2104.05695)

    Args:
        phi (float): rotation angle :math:`\phi`
        wires (Sequence[int]): the wires the operation acts on

    **Example**

    The following circuit performs the transformation :math:`|10\rangle\rightarrow \cos(
    \phi/2)|10\rangle -\sin(\phi/2)|01\rangle`:

    .. code-block::

        dev = qml.device('default.qubit', wires=2)

        @qml.qnode(dev)
        def circuit(phi):
            qml.PauliX(wires=0)
            qml.SingleExcitation(phi, wires=[0, 1])
            return qml.state()

        circuit(0.1)
    """

    num_params = 1
    num_wires = 2
    par_domain = "R"
    grad_method = "A"
    grad_recipe = four_term_grad_recipe
    generator = [
        np.array([[0, 0, 0, 0], [0, 0, -1j, 0], [0, 1j, 0, 0], [0, 0, 0, 0]]),
        -1 / 2,
    ]

    @classmethod
    def _matrix(cls, *params):
        theta = params[0]
        c = math.cos(theta / 2)
        s = math.sin(theta / 2)

        return np.array([[1, 0, 0, 0], [0, c, -s, 0], [0, s, c, 0], [0, 0, 0, 1]])

    @staticmethod
    def decomposition(theta, wires):
        decomp_ops = [
            qml.CNOT(wires=[wires[0], wires[1]]),
            qml.CRY(theta, wires=[wires[1], wires[0]]),
            qml.CNOT(wires=[wires[0], wires[1]]),
        ]
        return decomp_ops

    def adjoint(self):
        (phi,) = self.parameters
        return SingleExcitation(-phi, wires=self.wires)


class SingleExcitationMinus(Operation):
    r"""SingleExcitationMinus(phi, wires)
    Single excitation rotation with negative phase-shift outside the rotation subspace.

    .. math:: U_-(\phi) = \begin{bmatrix}
                e^{-i\phi/2} & 0 & 0 & 0 \\
                0 & \cos(\phi/2) & -\sin(\phi/2) & 0 \\
                0 & \sin(\phi/2) & \cos(\phi/2) & 0 \\
                0 & 0 & 0 & e^{-i\phi/2}
            \end{bmatrix}.

    **Details:**

    * Number of wires: 2
    * Number of parameters: 1
    * Gradient recipe: :math:`\frac{d}{d\phi}f(U_-(\phi)) = \frac{1}{2}\left[f(U_-(\phi+\pi/2)) - f(U_-(\phi-\pi/2))\right]`
      where :math:`f` is an expectation value depending on :math:`U_-(\phi)`.

    Args:
        phi (float): rotation angle :math:`\phi`
        wires (Sequence[int] or int): the wires the operation acts on

    """
    num_params = 1
    num_wires = 2
    par_domain = "R"
    grad_method = "A"
    generator = [
        np.array([[1, 0, 0, 0], [0, 0, -1j, 0], [0, 1j, 0, 0], [0, 0, 0, 1]]),
        -1 / 2,
    ]

    @classmethod
    def _matrix(cls, *params):
        theta = params[0]
        c = math.cos(theta / 2)
        s = math.sin(theta / 2)
        e = cmath.exp(-1j * theta / 2)

        return np.array([[e, 0, 0, 0], [0, c, -s, 0], [0, s, c, 0], [0, 0, 0, e]])

    @staticmethod
    def decomposition(theta, wires):
        decomp_ops = [
            qml.PauliX(wires=wires[0]),
            qml.PauliX(wires=wires[1]),
            qml.ControlledPhaseShift(-theta / 2, wires=[wires[1], wires[0]]),
            qml.PauliX(wires=wires[0]),
            qml.PauliX(wires=wires[1]),
            qml.ControlledPhaseShift(-theta / 2, wires=[wires[0], wires[1]]),
            qml.CNOT(wires=[wires[0], wires[1]]),
            qml.CRY(theta, wires=[wires[1], wires[0]]),
            qml.CNOT(wires=[wires[0], wires[1]]),
        ]
        return decomp_ops

    def adjoint(self):
        (phi,) = self.parameters
        return SingleExcitationMinus(-phi, wires=self.wires)


class SingleExcitationPlus(Operation):
    r"""SingleExcitationPlus(phi, wires)
    Single excitation rotation with positive phase-shift outside the rotation subspace.

    .. math:: U_+(\phi) = \begin{bmatrix}
                e^{i\phi/2} & 0 & 0 & 0 \\
                0 & \cos(\phi/2) & -\sin(\phi/2) & 0 \\
                0 & \sin(\phi/2) & \cos(\phi/2) & 0 \\
                0 & 0 & 0 & e^{i\phi/2}
            \end{bmatrix}.

    **Details:**

    * Number of wires: 2
    * Number of parameters: 1
    * Gradient recipe: :math:`\frac{d}{d\phi}f(U_+(\phi)) = \frac{1}{2}\left[f(U_+(\phi+\pi/2)) - f(U_+(\phi-\pi/2))\right]`
      where :math:`f` is an expectation value depending on :math:`U_+(\phi)`.

    Args:
        phi (float): rotation angle :math:`\phi`
        wires (Sequence[int] or int): the wires the operation acts on

    """
    num_params = 1
    num_wires = 2
    par_domain = "R"
    grad_method = "A"
    generator = [
        np.array([[-1, 0, 0, 0], [0, 0, -1j, 0], [0, 1j, 0, 0], [0, 0, 0, -1]]),
        -1 / 2,
    ]

    @classmethod
    def _matrix(cls, *params):
        theta = params[0]
        c = math.cos(theta / 2)
        s = math.sin(theta / 2)
        e = cmath.exp(1j * theta / 2)

        return np.array([[e, 0, 0, 0], [0, c, -s, 0], [0, s, c, 0], [0, 0, 0, e]])

    @staticmethod
    def decomposition(theta, wires):
        decomp_ops = [
            qml.PauliX(wires=wires[0]),
            qml.PauliX(wires=wires[1]),
            qml.ControlledPhaseShift(theta / 2, wires=[wires[1], wires[0]]),
            qml.PauliX(wires=wires[0]),
            qml.PauliX(wires=wires[1]),
            qml.ControlledPhaseShift(theta / 2, wires=[wires[0], wires[1]]),
            qml.CNOT(wires=[wires[0], wires[1]]),
            qml.CRY(theta, wires=[wires[1], wires[0]]),
            qml.CNOT(wires=[wires[0], wires[1]]),
        ]
        return decomp_ops

    def adjoint(self):
        (phi,) = self.parameters
        return SingleExcitationPlus(-phi, wires=self.wires)


class DoubleExcitation(Operation):
    r"""DoubleExcitation(phi, wires)
    Double excitation rotation.

    This operation performs an :math:`SO(2)` rotation in the two-dimensional subspace :math:`\{
    |1100\rangle,|0011\rangle\}`. More precisely, it performs the transformation

    .. math::

        &|0011\rangle \rightarrow \cos(\phi/2) |0011\rangle - \sin(\phi/2) |1100\rangle\\
        &|1100\rangle \rightarrow \cos(\phi/2) |1100\rangle + \sin(\phi/2) |0011\rangle,

    while leaving all other basis states unchanged.

    The name originates from the occupation-number representation of fermionic wavefunctions, where
    the transformation from :math:`|1100\rangle` to :math:`|0011\rangle` is interpreted as
    "exciting" two particles from the first pair of qubits to the second pair of qubits.

    **Details:**

    * Number of wires: 4
    * Number of parameters: 1
    * Gradient recipe: The ``DoubleExcitation`` operator satisfies a four-term parameter-shift rule
      (see Appendix F, https://arxiv.org/abs/2104.05695):

    Args:
        phi (float): rotation angle :math:`\phi`
        wires (Sequence[int]): the wires the operation acts on

    **Example**

    The following circuit performs the transformation :math:`|1100\rangle\rightarrow \cos(
    \phi/2)|1100\rangle +\sin(\phi/2)|0011\rangle)`:

    .. code-block::

        dev = qml.device('default.qubit', wires=4)

        @qml.qnode(dev)
        def circuit(phi):
            qml.PauliX(wires=0)
            qml.PauliX(wires=1)
            qml.DoubleExcitation(phi, wires=[0, 1, 2, 3])
            return qml.state()

        circuit(0.1)
    """

    num_params = 1
    num_wires = 4
    par_domain = "R"
    grad_method = "A"
    grad_recipe = four_term_grad_recipe

    G = np.zeros((16, 16), dtype=np.complex64)
    G[3, 12] = -1j  # 3 (dec) = 0011 (bin)
    G[12, 3] = 1j  # 12 (dec) = 1100 (bin)
    generator = [G, -1 / 2]

    @classmethod
    def _matrix(cls, *params):
        theta = params[0]
        c = math.cos(theta / 2)
        s = math.sin(theta / 2)

        U = np.eye(16)
        U[3, 3] = c  # 3 (dec) = 0011 (bin)
        U[3, 12] = -s  # 12 (dec) = 1100 (bin)
        U[12, 3] = s
        U[12, 12] = c

        return U

    @staticmethod
    def decomposition(theta, wires):
        # This decomposition is the "upside down" version of that on p17 of https://arxiv.org/abs/2104.05695
        decomp_ops = [
            qml.CNOT(wires=[wires[2], wires[3]]),
            qml.CNOT(wires=[wires[0], wires[2]]),
            qml.Hadamard(wires=wires[3]),
            qml.Hadamard(wires=wires[0]),
            qml.CNOT(wires=[wires[2], wires[3]]),
            qml.CNOT(wires=[wires[0], wires[1]]),
            qml.RY(theta / 8, wires=wires[1]),
            qml.RY(-theta / 8, wires=wires[0]),
            qml.CNOT(wires=[wires[0], wires[3]]),
            qml.Hadamard(wires=wires[3]),
            qml.CNOT(wires=[wires[3], wires[1]]),
            qml.RY(theta / 8, wires=wires[1]),
            qml.RY(-theta / 8, wires=wires[0]),
            qml.CNOT(wires=[wires[2], wires[1]]),
            qml.CNOT(wires=[wires[2], wires[0]]),
            qml.RY(-theta / 8, wires=wires[1]),
            qml.RY(theta / 8, wires=wires[0]),
            qml.CNOT(wires=[wires[3], wires[1]]),
            qml.Hadamard(wires=wires[3]),
            qml.CNOT(wires=[wires[0], wires[3]]),
            qml.RY(-theta / 8, wires=wires[1]),
            qml.RY(theta / 8, wires=wires[0]),
            qml.CNOT(wires=[wires[0], wires[1]]),
            qml.CNOT(wires=[wires[2], wires[0]]),
            qml.Hadamard(wires=wires[0]),
            qml.Hadamard(wires=wires[3]),
            qml.CNOT(wires=[wires[0], wires[2]]),
            qml.CNOT(wires=[wires[2], wires[3]]),
        ]

        return decomp_ops

    def adjoint(self):
        (theta,) = self.parameters
        return DoubleExcitation(-theta, wires=self.wires)


class DoubleExcitationPlus(Operation):
    r"""DoubleExcitationPlus(phi, wires)
    Double excitation rotation with positive phase-shift outside the rotation subspace.

    This operation performs an :math:`SO(2)` rotation in the two-dimensional subspace :math:`\{
    |1100\rangle,|0011\rangle\}` while applying a phase-shift on other states. More precisely,
    it performs the transformation

    .. math::

        &|0011\rangle \rightarrow \cos(\phi/2) |0011\rangle - \sin(\phi/2) |1100\rangle\\
        &|1100\rangle \rightarrow \cos(\phi/2) |1100\rangle + \sin(\phi/2) |0011\rangle\\
        &|x\rangle \rightarrow e^{i\phi/2} |x\rangle,

    for all other basis states :math:`|x\rangle`.

    **Details:**

    * Number of wires: 4
    * Number of parameters: 1
    * Gradient recipe: :math:`\frac{d}{d\phi}f(U_+(\phi)) = \frac{1}{2}\left[f(U_+(\phi+\pi/2)) - f(U_+(\phi-\pi/2))\right]`
      where :math:`f` is an expectation value depending on :math:`U_+(\phi)`

    Args:
        phi (float): rotation angle :math:`\phi`
        wires (Sequence[int]): the wires the operation acts on
    """

    num_params = 1
    num_wires = 4
    par_domain = "R"
    grad_method = "A"

    G = -1 * np.eye(16, dtype=np.complex64)
    G[3, 3] = 0
    G[12, 12] = 0
    G[3, 12] = -1j  # 3 (dec) = 0011 (bin)
    G[12, 3] = 1j  # 12 (dec) = 1100 (bin)
    generator = [G, -1 / 2]

    @classmethod
    def _matrix(cls, *params):
        theta = params[0]
        c = math.cos(theta / 2)
        s = math.sin(theta / 2)
        e = cmath.exp(1j * theta / 2)

        U = e * np.eye(16, dtype=np.complex64)
        U[3, 3] = c  # 3 (dec) = 0011 (bin)
        U[3, 12] = -s  # 12 (dec) = 1100 (bin)
        U[12, 3] = s
        U[12, 12] = c

        return U

    def adjoint(self):
        (theta,) = self.parameters
        return DoubleExcitationPlus(-theta, wires=self.wires)


class DoubleExcitationMinus(Operation):
    r"""DoubleExcitationMinus(phi, wires)
    Double excitation rotation with negative phase-shift outside the rotation subspace.

    This operation performs an :math:`SO(2)` rotation in the two-dimensional subspace :math:`\{
    |1100\rangle,|0011\rangle\}` while applying a phase-shift on other states. More precisely,
    it performs the transformation

    .. math::

        &|0011\rangle \rightarrow \cos(\phi/2) |0011\rangle - \sin(\phi/2) |1100\rangle\\
        &|1100\rangle \rightarrow \cos(\phi/2) |1100\rangle + \sin(\phi/2) |0011\rangle\\
        &|x\rangle \rightarrow e^{-i\phi/2} |x\rangle,

    for all other basis states :math:`|x\rangle`.

    **Details:**

    * Number of wires: 4
    * Number of parameters: 1
    * Gradient recipe: :math:`\frac{d}{d\phi}f(U_-(\phi)) = \frac{1}{2}\left[f(U_-(\phi+\pi/2)) - f(U_-(\phi-\pi/2))\right]`
      where :math:`f` is an expectation value depending on :math:`U_-(\phi)`

    Args:
        phi (float): rotation angle :math:`\phi`
        wires (Sequence[int]): the wires the operation acts on
    """

    num_params = 1
    num_wires = 4
    par_domain = "R"
    grad_method = "A"

    G = np.eye(16, dtype=np.complex64)
    G[3, 3] = 0
    G[12, 12] = 0
    G[3, 12] = -1j  # 3 (dec) = 0011 (bin)
    G[12, 3] = 1j  # 12 (dec) = 1100 (bin)
    generator = [G, -1 / 2]

    @classmethod
    def _matrix(cls, *params):
        theta = params[0]
        c = math.cos(theta / 2)
        s = math.sin(theta / 2)
        e = cmath.exp(-1j * theta / 2)

        U = e * np.eye(16, dtype=np.complex64)
        U[3, 3] = c  # 3 (dec) = 0011 (bin)
        U[3, 12] = -s  # 12 (dec) = 1100 (bin)
        U[12, 3] = s
        U[12, 12] = c

        return U

    def adjoint(self):
        (theta,) = self.parameters
        return DoubleExcitationMinus(-theta, wires=self.wires)


class OrbitalRotation(Operation):
    r"""OrbitalRotation(phi, wires)
    Spin-adapted spatial orbital rotation.

    For two neighbouring spatial orbitals :math:`\{|\Phi_{0}\rangle, |\Phi_{1}\rangle\}`, this operation
    performs the following transformation

    .. math::
        &|\Phi_{0}\rangle = \cos(\phi/2)|\Phi_{0}\rangle - \sin(\phi/2)|\Phi_{1}\rangle\\
        &|\Phi_{1}\rangle = \cos(\phi/2)|\Phi_{0}\rangle + \sin(\phi/2)|\Phi_{1}\rangle,

    with the same orbital operation applied in the :math:`\alpha` and :math:`\beta` spin orbitals.

    .. figure:: ../../_static/qchem/orbital_rotation_decomposition_extended.png
        :align: center
        :width: 100%
        :target: javascript:void(0);

    Here, :math:`G(\phi)` represents a single-excitation Givens rotation, implemented in PennyLane
    as the :class:`~.SingleExcitation` operation.

    **Details:**

    * Number of wires: 4
    * Number of parameters: 1
    * Gradient recipe: The ``OrbitalRotation`` operator satisfies the four-term parameter-shift rule
      (see Appendix F, https://arxiv.org/abs/2104.05695)

    Args:
        phi (float): rotation angle :math:`\phi`
        wires (Sequence[int]): the wires the operation acts on

    **Example**

    .. code-block::

        >>> dev = qml.device('default.qubit', wires=4)
        >>> @qml.qnode(dev)
        ... def circuit(phi):
        ...     qml.BasisState(np.array([1, 1, 0, 0]), wires=[0, 1, 2, 3])
        ...     qml.OrbitalRotation(phi, wires=[0, 1, 2, 3])
        ...     return qml.state()
        >>> circuit(0.1)
        array([ 0.        +0.j,  0.        +0.j,  0.        +0.j,
                0.00249792+0.j,  0.        +0.j,  0.        +0.j,
               -0.04991671+0.j,  0.        +0.j,  0.        +0.j,
               -0.04991671+0.j,  0.        +0.j,  0.        +0.j,
                0.99750208+0.j,  0.        +0.j,  0.        +0.j,
                0.        +0.j])
    """

    num_params = 1
    num_wires = 4
    par_domain = "R"
    grad_method = "A"
    grad_recipe = four_term_grad_recipe
    generator = [
        qml.math.array(
            [
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, -1j, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, -1j, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, -1j, 0, 0, -1j, 0, 0, 0, 0, 0, 0],
                [0, 1j, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1j, 0, 0, 0, 0, 0, 0, 0, 0, -1j, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1j, 0, 0],
                [0, 0, 1j, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1j, 0, 0, 0, 0, 0, 0, 0, 0, -1j, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1j, 0],
                [0, 0, 0, 0, 0, 0, 1j, 0, 0, 1j, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 1j, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1j, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ]
        ),
        -1 / 2,
    ]

    @classmethod
    def _matrix(cls, *params):
        # This matrix is the "sign flipped" version of that on p18 of https://arxiv.org/abs/2104.05695,
        # where the sign flip is to adjust for the opposite convention used by authors for naming wires.
        # Additionally, there was a typo in the sign of a matrix element "s" at [2, 8], which is fixed here.
        # matrix = qml.math.array(
        #     [
        #         [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        #         [0, c, 0, 0, -s, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        #         [0, 0, c, 0, 0, 0, 0, 0, -s, 0, 0, 0, 0, 0, 0, 0],
        #         [0, 0, 0, c ** 2, 0, 0, -c * s, 0, 0, -c * s, 0, 0, s ** 2, 0, 0, 0],
        #         [0, s, 0, 0, c, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        #         [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        #         [0, 0, 0, c * s, 0, 0, c ** 2, 0, 0, -(s ** 2), 0, 0, -c * s, 0, 0, 0],
        #         [0, 0, 0, 0, 0, 0, 0, c, 0, 0, 0, 0, 0, -s, 0, 0],
        #         [0, 0, s, 0, 0, 0, 0, 0, c, 0, 0, 0, 0, 0, 0, 0],
        #         [0, 0, 0, c * s, 0, 0, -(s ** 2), 0, 0, c ** 2, 0, 0, -c * s, 0, 0, 0],
        #         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        #         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, c, 0, 0, -s, 0],
        #         [0, 0, 0, s ** 2, 0, 0, c * s, 0, 0, c * s, 0, 0, c ** 2, 0, 0, 0],
        #         [0, 0, 0, 0, 0, 0, 0, s, 0, 0, 0, 0, 0, c, 0, 0],
        #         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, s, 0, 0, c, 0],
        #         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        #     ]
        # )

        # Determines the correct framework
        interface = qml.math._multi_dispatch(params)

        phi = params[0]
        c = qml.math.cos(phi / 2, like=interface)
        s = qml.math.sin(phi / 2, like=interface)

        # Determines the data type for casting other tensor-like objects
        dtype = c.dtype

        # Generate matrix with non-zero elements at indices where value is 1
        or_I4 = qml.math.array(
            qml.math.diag([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]),
            dtype=dtype,
            like=interface,
        )

        # Generate matrix with non-zero elements at indices where value is c
        or_cos = qml.math.array(
            qml.math.diag([0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0]),
            dtype=dtype,
            like=interface,
        )

        # Generate matrix with non-zero elements at indices where value is c**2
        or_cos_2 = qml.math.array(
            qml.math.diag([0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0]),
            dtype=dtype,
            like=interface,
        )

        # Generate matrix with non-zero elements at indices where value is s
        or_sin_array = qml.math.zeros((16, 16))
        rows, cols = [1, 2, 4, 7, 8, 11, 13, 14], [4, 8, 1, 13, 2, 14, 7, 11]
        for row, col in zip(rows, cols):
            or_sin_array[row, col] = qml.math.sign(row - col)  # if row < col: -1 else 1
        or_sin = qml.math.array(or_sin_array, dtype=dtype, like=interface)

        # Generate matrix with non-zero elements at indices where value is s**2
        or_sin_2 = qml.math.array(
            qml.math.fliplr(
                qml.math.diag([0, 0, 0, 1, 0, 0, -1, 0, 0, -1, 0, 0, 1, 0, 0, 0])
            ).copy(), # copy is necessary to remove negative stride introduced by fliplr
            dtype=s.dtype,
            like=interface,
        )

        # Generate matrix with non-zero elements at indices where value is c*s
        or_cos_sin_array = qml.math.zeros((16, 16))
        rows, cols = [3, 3, 6, 6, 9, 9, 12, 12], [6, 9, 3, 12, 3, 12, 6, 9]
        for row, col in zip(rows, cols):
            or_cos_sin_array[row, col] = qml.math.sign(row - col)  # if row < col: -1 else 1
        or_cos_sin = qml.math.array(or_cos_sin_array, dtype=dtype, like=interface)

        # Generate the complete gate matrix
        matrix = (
            or_I4
            + c * or_cos
            + (c ** 2) * or_cos_2
            + s * or_sin
            + (s ** 2) * or_sin_2
            + c * s * or_cos_sin
        )

        return matrix

    @staticmethod
    def decomposition(phi, wires):
        # This decomposition is the "upside down" version of that on p18 of https://arxiv.org/abs/2104.05695
        decomp_ops = [
            qml.Hadamard(wires=wires[3]),
            qml.Hadamard(wires=wires[2]),
            qml.CNOT(wires=[wires[3], wires[1]]),
            qml.CNOT(wires=[wires[2], wires[0]]),
            qml.RY(phi / 2, wires=wires[3]),
            qml.RY(phi / 2, wires=wires[2]),
            qml.RY(phi / 2, wires=wires[1]),
            qml.RY(phi / 2, wires=wires[0]),
            qml.CNOT(wires=[wires[3], wires[1]]),
            qml.CNOT(wires=[wires[2], wires[0]]),
            qml.Hadamard(wires=wires[3]),
            qml.Hadamard(wires=wires[2]),
        ]
        return decomp_ops

    def adjoint(self):
        (phi,) = self.parameters
        return OrbitalRotation(-phi, wires=self.wires)
