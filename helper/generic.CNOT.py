import numpy as np

def generate_cnot_matrix(num_qubits, control, target):
    """
    Generates the CNOT matrix for a given number of qubits,
    with a specified control and target qubit.
    
    Parameters:
        num_qubits (int): Total number of qubits in the system.
        control (int): The index of the control qubit (0-based).
        target (int): The index of the target qubit (0-based).
    
    Returns:
        np.ndarray: The resulting CNOT matrix of size (2^num_qubits, 2^num_qubits).
    """
    if(control>=num_qubits):
        raise Exception(f"CircuitError: 'Index {control} out of range for size {num_qubits}.'")
    if(target>=num_qubits):
        raise Exception(f"CircuitError: 'Index {target} out of range for size {num_qubits}.'")
    if(control<0):
        raise Exception(f"CircuitError: 'Index {control} out of range for size 0.'")
    if(target<0):
        raise Exception(f"CircuitError: 'Index {target} out of range for size 0.'")
    
    size = 2 ** num_qubits  # Total dimension of the Hilbert space
    cnot_matrix = np.eye(size)  # Start with an identity matrix
    
    for i in range(size):
        binary_i = format(i, f'0{num_qubits}b')  # Convert index to binary representation
        if binary_i[num_qubits - 1 - control] == '1':  # If control qubit is 1
            # Flip the target qubit
            flipped_i = list(binary_i)
            flipped_i[num_qubits - 1 - target] = '1' if flipped_i[num_qubits - 1 - target] == '0' else '0'
            flipped_index = int("".join(flipped_i), 2)  # Convert back to integer index
            
            # Swap rows i and flipped_index in the matrix
            cnot_matrix[i, i] = 0
            cnot_matrix[flipped_index, flipped_index] = 0
            cnot_matrix[i, flipped_index] = 1
            cnot_matrix[flipped_index, i] = 1
    
    return cnot_matrix

# Example usage
num_qubits = 4
control = 40
target = 1
cnot_matrix = generate_cnot_matrix(num_qubits, control, target)
print(f"CNOT Matrix for control q{control} and target q{target}:")
print(cnot_matrix)

# CITATION 
# https://cnot.io/quantum_computing/two_qubit_operations.html
# https://quantumcomputing.stackexchange.com/questions/17599/how-to-represent-a-cnot-gate-operating-on-three-qubit-states-as-a-matrix