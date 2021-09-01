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
This file automatically generates and saves a series of example pictures for the 
circuit drawer.  This will be useful during early stages when the project is still
undergoing cosmetic changes.
"""

import matplotlib.pyplot as plt

from pennylane.circuit_drawer import MPLDrawer
from os import path

folder = path.dirname(path.realpath(__file__))

def labels(savefile="labels.png"):
    drawer = MPLDrawer(n_wires=2, n_layers=1)
    drawer.label(["a", "b"])
    plt.savefig(path.join(folder, savefile))
    plt.close()

def labels_formatted(savefile="labels_formatted.png"):
    drawer = MPLDrawer(n_wires=2, n_layers=1)
    drawer.label(["a", "b"], text_kwargs={"color": "indigo", "fontsize": "xx-large"})
    plt.savefig(path.join(folder, savefile))
    plt.close()

def box_gates(savefile="box_gates.png"):
    drawer = MPLDrawer(n_wires=2, n_layers=1)

    drawer.box_gate(layer=0, wires=(0, 1), text="CY")
    plt.savefig(path.join(folder, savefile))
    plt.close()

def box_gates_formatted(savefile="box_gates_formatted.png"):
    box_kwargs = {'facecolor':'lightcoral', 'edgecolor': 'maroon', 'linewidth': 5}
    text_kwargs = {'fontsize': 'xx-large', 'color':'maroon'}

    drawer = MPLDrawer(n_wires=2, n_layers=1)

    drawer.box_gate(layer=0, wires=(0, 1), text="CY", 
        box_kwargs=box_kwargs, text_kwargs=text_kwargs)
    plt.savefig(path.join(folder, savefile))
    plt.close()

def ctrl(savefile="ctrl.png"):
    drawer = MPLDrawer(n_wires=2, n_layers=3)

    drawer.ctrl(layer=0, wire_ctrl=0, wire_target=1)
    drawer.ctrl(layer=1, wire_ctrl=(0,1), control_values=[0,1])
    drawer.ctrl(layer=2, wire_ctrl=(0,1), color="indigo")
    plt.savefig(path.join(folder, savefile))
    plt.close()

def CNOT(savefile="cnot.png"):
    drawer = MPLDrawer(n_wires=2, n_layers=2)

    drawer.CNOT(0, (0, 1))
    drawer.CNOT(1, (1, 0), color='indigo')
    plt.savefig(path.join(folder, savefile))
    plt.close()

def SWAP(savefile="SWAP.png"):
    drawer = MPLDrawer(n_wires=2, n_layers=1)

    drawer.SWAP(0, (0, 1))
    plt.savefig(path.join(folder, savefile))
    plt.close()

def SWAP_formatted(savefile="SWAP_formatted.png"):
    swap_keywords = {"linewidth":2, "color":"indigo"}

    drawer = MPLDrawer(n_wires=2, n_layers=1)

    drawer.SWAP(0, (0, 1), kwargs=swap_keywords)
    plt.savefig(path.join(folder, savefile))
    plt.close()

def measure(savefile="measure.png"):
    drawer = MPLDrawer(n_wires=1, n_layers=1)

    drawer.measure(0, 0)
    plt.savefig(path.join(folder, savefile))
    plt.close()

def measure_formatted(savefile="measure_formatted.png"):
    drawer = MPLDrawer(n_wires=1, n_layers=1)

    measure_box = {'facecolor': 'white', 'edgecolor': 'indigo'}
    measure_lines = {'edgecolor': 'indigo', 'facecolor': 'plum', 'linewidth': 2}
    drawer.measure(0, 0, box_kwargs=measure_box, lines_kwargs=measure_lines)
    plt.savefig(path.join(folder, savefile))
    plt.close()

def integration(style='default', savefile="example_basic.png"):
    plt.style.use(style)
    drawer = MPLDrawer(n_wires=5,n_layers=5)

    drawer.label(["0","a",r"$|\Psi\rangle$",r"$|\theta\rangle$", "aux"])

    drawer.box_gate(0, [0,1,2,3,4], "Entangling Layers", text_kwargs={'rotation':'vertical'})
    drawer.box_gate(1, [0, 1], "U(θ)")

    drawer.box_gate(1, 4, "Z")

    drawer.SWAP(1, (2, 3))
    drawer.CNOT(2, (0,2))

    drawer.ctrl(3, [1,3], control_values = [True, False])
    drawer.box_gate(3, 2, "H", zorder=2)

    drawer.ctrl(4, [1,2])

    drawer.measure(5, 0)

    drawer.fig.suptitle('My Circuit', fontsize='xx-large')
    plt.savefig(path.join(folder, savefile))
    plt.style.use('default')
    plt.close()

def integration_rcParams(savefile="example_rcParams.png"):
    plt.rcParams['patch.facecolor'] = 'white'
    plt.rcParams['patch.edgecolor'] = 'black'
    plt.rcParams['patch.linewidth'] = 2
    plt.rcParams['patch.force_edgecolor'] = True

    plt.rcParams['lines.color'] = 'black'

    drawer = MPLDrawer(n_wires=5,n_layers=5)

    drawer.label(["0","a",r"$|\Psi\rangle$",r"$|\theta\rangle$", "aux"])

    drawer.box_gate(0, [0,1,2,3,4], "Entangling Layers", text_kwargs={'rotation':'vertical'})
    drawer.box_gate(1, [0, 1], "U(θ)")

    drawer.box_gate(1, 4, "Z")

    drawer.SWAP(1, (2, 3))
    drawer.CNOT(2, (0,2))

    drawer.ctrl(3, [1,3], control_values = [True, False])
    drawer.box_gate(3, 2, "H", zorder=2)

    drawer.ctrl(4, [1,2])

    drawer.measure(5, 0)

    drawer.fig.suptitle('My Circuit', fontsize='xx-large')


    plt.savefig(path.join(folder, savefile))
    plt.style.use('default')
    plt.close()

def integration_formatted(savefile="example_formatted.png"):

    wire_kwargs = {"color": "indigo", "linewidth": 4}
    label_kwargs = {"fontsize": "x-large", 'color': 'indigo'}
    box_kwargs = {'facecolor':'lightcoral', 'edgecolor': 'maroon', 'linewidth': 5}
    text_kwargs = {'fontsize': 'xx-large', 'color':'maroon'}
    swap_kwargs = {'linewidth': 4, 'color': 'darkgreen'}

    measure_box = {'facecolor': 'white', 'edgecolor': 'indigo'}
    measure_lines = {'edgecolor': 'indigo', 'facecolor': 'plum', 'linewidth': 2}

    #########################################################

    drawer = MPLDrawer(n_wires=2,n_layers=4, wire_kwargs=wire_kwargs)

    drawer.label(["0","a"],
                text_kwargs=label_kwargs)

    drawer.box_gate(0, 0, "Z", box_kwargs = box_kwargs, text_kwargs=text_kwargs)

    drawer.SWAP(1, (0,1), kwargs=swap_kwargs)

    drawer.CNOT(2, (0,1), color='teal')

    drawer.ctrl(3, (0,1), color='black')

    for wire in range(2):
        drawer.measure(4, wire, box_kwargs=measure_box, lines_kwargs=measure_lines)

    drawer.fig.suptitle('My Circuit', fontsize='xx-large')
    plt.savefig(path.join(folder, savefile))
    plt.close()

def float_layer(savefile="float_layer.png"):

    drawer = MPLDrawer(2,2)

    drawer.box_gate(layer=0.5, wires=0, text="Big Gate", extra_width=0.5)
    drawer.box_gate(layer=0, wires=1, text="X")
    drawer.box_gate(layer=1, wires=1, text="Y")

    plt.savefig(path.join(folder, savefile))
    plt.close()


if __name__ == "__main__":
    labels()
    labels_formatted()
    box_gates()
    box_gates_formatted()
    ctrl()
    CNOT()
    SWAP()
    SWAP_formatted()
    measure()
    measure_formatted()
    integration()
    float_layer()
    integration(style='Solarize_Light2', savefile="example_Solarize_Light2.png")
    integration_rcParams()
    integration_formatted()

