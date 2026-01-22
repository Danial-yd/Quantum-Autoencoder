from qiskit import QuantumCircuit
from qiskit.visualization import plot_histogram

qc_output = QuantumCircuit(8)
qc_output.measure_all()

qc_output.draw(initial_state=True)

from qiskit import QuantumCircuit, transpile
from qiskit_aer import Aer
from qiskit.visualization import plot_histogram, plot_bloch_vector
from math import sqrt, pi

qc = QuantumCircuit(1)  # Create a quantum circuit with one qubit
initial_state = [0,1]   # Define initial_state as |1>
qc.initialize(initial_state, 0) # Apply initialisation operation to the 0th qubit
qc.draw()

sim = Aer.get_backend('aer_simulator')

qc = QuantumCircuit(1)  # Create a quantum circuit with one qubit
initial_state = [0,1]   # Define initial_state as |1>
qc.initialize(initial_state, 0) # Apply initialisation operation to the 0th qubit
qc.save_statevector()   # Tell simulator to save statevector
qobj = transpile(qc)     # Create a Qobj from the circuit for the simulator to run
result = sim.run(qobj).result()

out_state = result.get_statevector()
print(out_state)

import numpy as np
from qiskit.visualization import plot_histogram, plot_bloch_multivector

qc = QuantumCircuit(3)
# Apply H-gate to each qubit:
for qubit in range(3):
    qc.h(qubit)
# See the circuit:
qc.draw()


qc = QuantumCircuit(2)
qc.h(0)
qc.x(1)
qc.draw()


usim = Aer.get_backend('aer_simulator')
qc.save_unitary()
qobj = transpile(qc)
unitary = usim.run(qobj).result().get_unitary()

from qiskit.visualization import array_to_latex
array_to_latex(unitary, prefix="\\text{Circuit = }\n")


qc = QuantumCircuit(2)
# Apply H-gate to the first:
qc.h(0)
qc.draw()

svsim = Aer.get_backend('aer_simulator')
qc.save_statevector()
qobj = transpile(qc)
final_state = svsim.run(qobj).result().get_statevector()
from qiskit.quantum_info import Statevector
from qiskit.visualization import array_to_latex
import sys, pkgutil
array_to_latex(final_state, prefix="\\text{Statevector = }")
print("Python exe:", sys.executable)
import sympy
print("sympy version:", sympy.__version__)
final_state = np.array([1/np.sqrt(2), 0, 0, 1/np.sqrt(2)])
latex_str = array_to_latex(final_state, prefix="\\text{Statevector = }")