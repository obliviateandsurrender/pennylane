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
r"""
Contains the QuantumNumberPreserving Gate Fabric template.
"""
# pylint: disable-msg=too-many-branches,too-many-arguments,protected-access
import numpy as np
import pennylane as qml
from pennylane.operation import Operation, AnyWires


class QuantumNumberPreservingU2(Operation):
    r"""Implements a local, expressive, and quantum-number-preserving ansatz using VQE circuit fabrics
    proposed by Anselmetti *et al.* in `arXiv:2104.05692 <https://arxiv.org/abs/2104.05692>`_.

    This template prepares the :math:`2M` qubits trial state, where `M` is the number of spatial orbitals.
    It uses 4-local-nearest-neighbor-tessellation of alternating even and off spatial-orbitals-pair 2-parameter,
    4-qubit :math:`\hat{Q}(\varphi, \theta)` gates. Each :math:`\hat{Q}` gate consists of a 1-parameter, 4-qubit spatial rotation
    gate :math:`\text{QuantumNumberPreserving}_{OR}(\varphi)`, and  a 1-parameter, 4-qubit diagonal pair exchange gate :math:`\text{DoubleExcitation}(\theta)`
    gate. In the :math:`\hat{Q}` gate we also allow inclusion of an optional constant :math:`\hat{\Pi}` gate,
    with choices :math:`\hat{\Pi} \in \{\hat{I}, \text{QuantumNumberPreserving}_{OR}(\pi)\}`.

    The circuit implementing the gate fabric layer for `M = 4` is shown below:

    |

    .. figure:: ../../_static/templates/layers/quantum_number_preserving_u2.png
        :align: center
        :width: 25%
        :target: javascript:void(0);

    |

    The 2-parameter, 4-qubit :math:`\hat{Q}` gate is decomposed as follows:

    |

    .. figure:: ../../_static/templates/layers/quantum_number_preserving_decompos.png
        :align: center
        :width: 100%
        :target: javascript:void(0);

    |

    Args:
        weights (tensor_like): Array of weights of shape ``(L, D, 2)``\.
            ``L`` is the number of gate fabric layers and :math:`D=M-1`
            is the number of :math:`\hat{Q}(\varphi, \theta)` gates per layer.
        wires (Iterable): wires that the template acts on
        init_state (tensor_like): iterable or shape ``(len(wires),)`` tensor representing the Hartree-Fock state
            used to initialize the wires
        pi_gate_include (boolean): If `True`, sets the optional constant :math:`\Pi` to :math:`\text{QuantumNumberPreserving}_{OR}(\pi)`.
            Default value is :math:`\hat{I}`.

    .. UsageDetails::

        #. The number of wires :math:`N` has to be equal to the number of
           spin orbitals included in the active space.

        #. The number of trainable parameters scales linearly with the number of layers as
           :math:`2D(N-1)`.

        An example of how to use this template is shown below:

        .. code-block:: python

            import pennylane as qml
            from pennylane.templates import ParticleConservingU1
            from functools import partial

            # Build the electronic Hamiltonian from a local .xyz file
            h, qubits = qml.qchem.molecular_hamiltonian("h2", "h2.xyz")

            # Define the Hartree-Fock state
            electrons = 2
            ref_state = qml.qchem.hf_state(electrons, qubits)

            # Define the device
            dev = qml.device('default.qubit', wires=qubits)

            # Define the ansatz
            ansatz = partial(QuantumNumberPreservingU2, init_state=ref_state, pi_gate_include=True)

            # Define the cost function
            cost_fn = qml.ExpvalCost(ansatz, h, dev)

            # Compute the expectation value of 'h'
            layers = 2
            params = qml.init.quantum_number_preserving_u2_normal(layers, qubits)
            print(cost_fn(params))

        **Parameter shape**

        The shape of the weights argument can be computed by the static method
        :meth:`~.QuantumNumberPreservingU2.shape` and used when creating randomly
        initialised weight tensors:

        .. code-block:: python

            shape = QuantumNumberPreservingU2.shape(n_layers=2, n_wires=2)
            weights = np.random.random(size=shape)


    """
    num_params = 1
    num_wires = AnyWires
    par_domain = "A"

    def __init__(
        self, weights, wires=None, init_state=None, pi_gate_include=False, do_queue=True, id=None
    ):

        if len(wires) < 4:
            raise ValueError(
                "This template requires the number of qubits to be greater than four; got {}".format(
                    len(wires)
                )
            )
        if len(wires) % 2:
            raise ValueError(
                "This template requires the number of qubits to be multiple of 2; got {}".format(
                    len(wires)
                )
            )

        self.M = len(wires) // 2

        self.qwires = [
            wires[i : i + 4] for i in range(0, len(wires), 4) if len(wires[i : i + 4]) == 4
        ]
        if self.M > 2:
            self.qwires += [
                wires[i : i + 4] for i in range(2, len(wires), 4) if len(wires[i : i + 4]) == 4
            ]

        shape = qml.math.shape(weights)

        if len(shape) != 3:
            raise ValueError(f"Weights tensor must be 3-dimensional; got shape {shape}")

        if shape[1] != len(self.qwires):
            raise ValueError(
                f"Weights tensor must have second dimension of length {len(self.qwires)}; got {shape[1]}"
            )

        if shape[2] != 2:
            raise ValueError(
                f"Weights tensor must have third dimension of length 2; got {shape[2]}"
            )

        self.n_layers = shape[0]
        # we can extract the numpy representation here
        # since init_state can never be differentiable
        if init_state is None:
            raise ValueError(f"Inital state should be provided; got {init_state}")
        self.init_state = qml.math.toarray(init_state)

        self.pi_gate = pi_gate_include

        super().__init__(weights, wires=wires, do_queue=do_queue, id=id)

    def expand(self):

        with qml.tape.QuantumTape() as tape:

            qml.templates.BasisEmbedding(self.init_state, wires=self.wires)
            weight = self.parameters[0]

            for layer in range(self.n_layers):
                for idx, wires in enumerate(self.qwires):

                    if self.pi_gate:
                        qml.QuantumNumberPreservingOR(np.pi, wires=wires)

                    qml.DoubleExcitation(weight[layer][idx][0], wires=wires)
                    qml.QuantumNumberPreservingOR(weight[layer][idx][1], wires=wires)

        return tape

    @staticmethod
    def shape(n_layers, n_wires):
        r"""Returns the shape of the weight tensor required for this template.

        Args:
            n_layers (int): number of layers
            n_wires (int): number of qubits

        Returns:
            tuple[int]: shape
        """

        if n_wires < 4:
            raise ValueError(
                "This template requires the number of qubits to be greater than four; got {}".format(
                    n_wires
                )
            )
        if n_wires % 2:
            raise ValueError(
                "This template requires the number of qubits to be multiple of 2; got {}".format(
                    n_wires
                )
            )

        return n_layers, n_wires // 2 - 1, 2