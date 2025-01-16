#   pip install qiskit[visualization]
import numpy as np

from qiskit import QuantumCircuit
import qiskit.quantum_info as qi

circuit = QuantumCircuit(4)
circuit.cx(4,1)
op = qi.Operator(circuit)
print(np.real(op) ) #since the matrix has no complex coeff, this makes it easier to look at 
op.draw()
