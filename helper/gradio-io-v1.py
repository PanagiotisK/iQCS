import gradio as gr
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import base64

def hadamard():
    return np.array([[1, 1], [1, -1]]) / np.sqrt(2)

def pauli_x():
    return np.array([[0, 1], [1, 0]])

def pauli_y():
    return np.array([[0, -1j], [1j, 0]])

def pauli_z():
    return np.array([[1, 0], [0, -1]])

def identity():
    return np.eye(2)

def cnot():
    return np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])

def toffoli():
    return np.array([
        [1, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 1, 0]
    ])

def apply_gate_to_qubit(gate, state, qubit, num_qubits):
    try:
        full_operator = 1
        for i in range(num_qubits):
            if i == qubit:
                full_operator = np.kron(full_operator, gate)
            else:
                full_operator = np.kron(full_operator, identity())
        return np.dot(full_operator, state)
    except Exception as e:
        raise ValueError(f"Error applying gate: {str(e)}")

def process_quantum_instructions(instructions: str, debug: bool = False):
    error_message = ""
    debug_info = ""
    try:
        lines = instructions.split("\n")
        num_qubits = max([int(parts[1][1:]) for line in lines if (parts := line.strip().split()) and len(parts) >= 2]) + 1
        initial_state = np.zeros(2**num_qubits, dtype=complex)
        initial_state[0] = 1  # |00...0>
        final_state = initial_state

        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 2:
                gate, qubit = parts[0].lower(), int(parts[1][1:])
                if gate == "h":
                    final_state = apply_gate_to_qubit(hadamard(), final_state, qubit, num_qubits)
                elif gate == "x":
                    final_state = apply_gate_to_qubit(pauli_x(), final_state, qubit, num_qubits)
                elif gate == "y":
                    final_state = apply_gate_to_qubit(pauli_y(), final_state, qubit, num_qubits)
                elif gate == "z":
                    final_state = apply_gate_to_qubit(pauli_z(), final_state, qubit, num_qubits)
                else:
                    error_message += f"Invalid instruction: {line}\n"
                
                if debug:
                    debug_info += f"After {line}:\n{final_state}\n"
    
        # Visualization
        fig, ax = plt.subplots()
        ax.bar([f"|{bin(i)[2:].zfill(num_qubits)}\u27E9" for i in range(2**num_qubits)], np.abs(final_state) ** 2)
        ax.set_ylabel("Probability")
        buf = BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        encoded_img = base64.b64encode(buf.getvalue()).decode('utf-8')
        img_html = f'<img src="data:image/png;base64,{encoded_img}" width="400">'
    
        return str(final_state), img_html, error_message, debug_info
    except Exception as e:
        return "", "", f"Error: {str(e)}", ""

iface = gr.Interface(
    fn=process_quantum_instructions,
    inputs=[
        gr.Textbox(lines=10, placeholder="Enter quantum instructions (e.g., H q0, CX q0 q1)"),
        gr.Checkbox(label="Enable Debug Mode")
    ],
    outputs=[
        gr.Textbox(label="Quantum State"),
        gr.HTML(label="State Visualization"),
        gr.Textbox(label="Errors", interactive=False),
        gr.Textbox(label="Debug Info", interactive=False)
    ],
    title="Custom Quantum Processor Simulator",
    description="Enter instructions to manipulate a quantum circuit and visualize the state."
)

if __name__ == "__main__":
    iface.launch(debug=True)
