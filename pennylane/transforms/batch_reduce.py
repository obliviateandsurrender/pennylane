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
"""Contains transforms for registering batch reductions of tapes and QNodes."""
# pylint: disable=too-few-public-methods
from collections.abc import Sequence
import functools
import inspect
import warnings

import pennylane as qml


def batch_tape_execute(tapes, device, batch_execute=False, parallel=False, **kwargs):
    """Execute a sequence of independent tapes on a device or devices.

    This function helps automate the process of:

    - Submitting a batch of tapes within a single job (if the device supports
      the ``batch_execute`` method), or
    - Executing a batch of tapes in parallel across multiple devices.

    Args:
        tapes (Sequence[.QuantumTape]): sequence of independent tapes to execute
        device (.Device or Sequence[.Device]): Corresponding device(s) where the tapes
            should be executed. This can either be a single device, or a list
            of devices of length ``len(tapes)``.
        batch_execute (bool): If set to ``True``, the batch of tapes will
            be executed via the QNode devices' ``batch_execute`` method. This can see significant
            performance improvements on remote simulator and hardware devices. Note that
            batch execution currently **does not support differentiability**.
        parallel (bool): If set to ``True``, the tapes are executed in parallel
            using the `Dask <https://dask.org/>`_ parallelism library.
            Note that the ``batch_execute`` and ``parallel`` keyword arguments are **mutually exclusive**,
            and cannot both be set to ``True``.

    Returns:
        list[float]: a one-dimensional list containing the numerical results corresponding
        to each tape execution

    **Example**

    Consider the following list of tapes:

    .. code-block:: python

        tapes = []

        for x in [0.1, 0.2, 0.3]:
            with qml.tape.QuantumTape() as tape:
                qml.Hadamard(wires=0)
                qml.CNOT(wires=[0, 1])
                qml.RX(x, wires=0)
                qml.probs(wires=[0, 1])

            tapes.append(tape)

    We can execute these tapes in a batch using ``batch_tape_execute``:

    >>> dev = qml.device("default.qubit", wires=2)
    >>> qml.transforms.batch_tape_execute(tapes, dev)
    [array([[0.49875104, 0.00124896, 0.00124896, 0.49875104]]),
     array([[0.49501664, 0.00498336, 0.00498336, 0.49501664]]),
     array([[0.48883412, 0.01116588, 0.01116588, 0.48883412]])]
    """
    if batch_execute and parallel:
        raise ValueError("'batch_execute' and 'parallel' cannot both be set to True.")

    if batch_execute:
        if isinstance(device, Sequence):
            raise ValueError("'batch_execute=True' is only supported for a single device.")

        warnings.warn(
            "'batch_execute=True' currently does not support differentiability.", UserWarning
        )

        return device.batch_execute(tapes)

    if not isinstance(device, Sequence):
        # broadcast the single device over all tapes
        device = [device] * len(tapes)

    if parallel:
        try:
            import dask  # pylint: disable=import-outside-toplevel
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "Dask must be installed for parallel evaluation. "
                "\nDask can be installed using pip:"
                "\n\npip install dask[delayed]"
            ) from e

        if not isinstance(device, Sequence):
            # broadcast the single device over all tapes
            device = [device] * len(tapes)

        results = []

        for t, d in zip(tapes, device):
            results.append(dask.delayed(t.execute)(device=d))

        _scheduler = kwargs.pop("scheduler", "threads")
        return dask.compute(*results, scheduler=_scheduler)

    return [t.execute(device=d) for t, d in zip(tapes, device)]


def _create_batch_reduce_internal_wrapper(fn, qnode, transform_args, transform_kwargs):
    """Convenience function to create the internal wrapper function
    generated by the batch_reduce decorator"""

    if not isinstance(qnode, qml.QNode):
        raise ValueError(
            f"The object to transform, {qnode}, does not appear " "to be a valid QNode."
        )

    parallel = transform_kwargs.pop("parallel", False)
    batch_execute = transform_kwargs.pop("batch_execute", False)

    @functools.wraps(qnode)
    def internal_wrapper(*args, **kwargs):
        qnode.construct(args, kwargs)

        if isinstance(fn, qml.single_tape_transform):
            reduction_fn = None
            tapes = fn(qnode.qtape, *transform_args, **transform_kwargs)
        else:
            tapes, reduction_fn = fn(qnode, *transform_args, **transform_kwargs)

        if not isinstance(tapes, Sequence):
            # quantum function returned a single tape
            tapes = [tapes]

        if reduction_fn is None:
            reduction_fn = lambda x: x

        res = batch_tape_execute(
            tapes, qnode.device, batch_execute=batch_execute, parallel=parallel
        )

        return reduction_fn(res)

    internal_wrapper.qnode = qnode
    internal_wrapper.interface = qnode.interface
    internal_wrapper.device = qnode.device
    return internal_wrapper


def batch_reduce(fn):
    """Register a new batch reduce transform.

    A batch reduction transform takes a QNode as input, and:

    - Generates a list of *independent* quantum tapes that can be executed in a batch,
    - Provides a reduction function to be applied to the results of the
      batch executed quantum tapes.

    Args:
        fn (callable): the batch reduction transform to register.

            Allowed batch reductions must be functions of the following form:

            .. code-block:: python

                def my_custom_reduction(qnode, *args, **kwargs):
                    ...
                    return tapes, processing_fn

            That is, the first argument must be the input QNode to transform,
            and the function must return a tuple ``(list, function)`` containing:

            * A list of new tapes to execute in a batch, and
            * A processing function with signature ``processing_fn(List[float])``
              which is applied to the flat list of results from the executed tapes.

            If ``tapes`` is empty, then it is assumed no quantum evaluations
            are required, and ``processing_fn`` will be passed an empty list.

    Returns:
        function: A hybrid quantum-classical function. Takes the same input arguments as
        the input QNode, as well as two additional **experimental** keyword arguments:

        * **batch_execute=False** (``bool``): If set to ``True``, the batch of tapes will
          be executed via the QNode devices' ``batch_execute`` method. This can see significant
          performance improvements on remote simulator and hardware devices. Note that
          batch execution currently **does not support differentiability**.

        * **parallel=False** (``bool``): If set to ``True``, the tapes are executed in parallel
          using the `Dask <https://dask.org/>`_ parallelism library.

        Note that the ``batch_execute`` and ``parallel`` keyword arguments are **mutually exclusive**,
        and cannot both be set to ``True``.

    **Example**

    Given a simple tape transform, that replaces the :class:`~.CRX` gate with a
    :class:`~.RY` and :class:`~.CZ` operation,

    .. code-block:: python

        @qml.single_tape_transform
        def tape_transform(tape, x):
            for op in tape.operations + tape.measurements:
                if op.name == "CRX":
                    qml.RY(op.parameters[0] * qml.math.sqrt(x), wires=op.wires[1])
                    qml.CZ(wires=op.wires)
                else:
                    op.queue()

    Let's build a QNode batch reduction that applies this transform twice with different
    transform parameters to create two independent tapes that can be batch executed,
    and then sums the results:

    .. code-block:: python

        @qml.batch_reduce
        def my_transform(qnode, x, y):
            tape1 = tape_transform(qnode.qtape, x)
            tape2 = tape_transform(qnode.qtape, y)

            def processing_fn(results):
                return qml.math.sum(results)

            return [tape1, tape2], processing_fn

    It can then be used to transform an existing QNode:

    .. code-block:: python

        dev = qml.device("default.qubit", wires=2)

        @my_transform(0.7, 0.8)
        @qml.qnode(dev)
        def circuit(x):
            qml.Hadamard(wires=0)
            qml.CRX(x, wires=[0, 1])
            return qml.expval(qml.PauliZ(1))

    >>> circuit(0.6)
    1.7360468658221193

    Not only is the transformed QNode fully differentiable, but the QNode transform
    parameters *themselves* are differentiable:

    .. code-block:: python

        @qml.qnode(dev)
        def circuit(x):
            qml.Hadamard(wires=0)
            qml.CRX(x, wires=[0, 1])
            return qml.expval(qml.PauliZ(1))

        def cost_fn(x, transform_weights):
            transform_fn = my_transform(*transform_weights)(circuit)
            return transform_fn(x)

    Evaluating the transform, as well as the derivative, with respect to the gate
    parameter *and* the transform weights:

    >>> x = np.array(0.6, requires_grad=True)
    >>> transform_weights = np.array([0.7, 0.8], requires_grad=True)
    >>> cost_fn(x, transform_weights)
    1.7360468658221193
    >>> qml.grad(cost_fn)(x, transform_weights)
    (array(-0.85987045), array([-0.17253469, -0.17148357]))
    """
    if not callable(fn):
        raise ValueError(
            "The qnode_transform decorator can only be applied "
            "to valid Python functions or callables."
        )

    sig = inspect.signature(fn)
    params = sig.parameters

    if len(params) > 1:

        @functools.wraps(fn)
        def make_batch_reduce(*targs, **tkwargs):
            def wrapper(qnode):
                return _create_batch_reduce_internal_wrapper(fn, qnode, targs, tkwargs)

            return wrapper

    elif len(params) == 1:

        @functools.wraps(fn)
        def make_batch_reduce(qnode):
            return _create_batch_reduce_internal_wrapper(fn, qnode, [], {})

    return make_batch_reduce
