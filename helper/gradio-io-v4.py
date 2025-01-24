import gradio as gr
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import re

def hadamard():
    return np.array([[1, 1], [1, -1]]) / np.sqrt(2)

def pauli_x():
    return np.array([[0, 1], [1, 0]])

def pauli_y():
    return np.array([[0, -1j], [1j, 0]])

def pauli_z():
    return np.array([[1, 0], [0, -1]])

def identity(size=2):
    return np.eye(size)

def validate_qubit_format(qubit_str):
    return re.fullmatch(r'q\d+', qubit_str) is not None

def apply_single_qubit_gate(gate, state, qubit, num_qubits):
    try:
        full_operator = np.eye(1, dtype=complex)
        for i in range(num_qubits):
            if i == qubit:
                full_operator = np.kron(full_operator, gate)
            else:
                full_operator = np.kron(full_operator, identity())
        return np.dot(full_operator, state)
    except Exception as e:
        raise ValueError(f"Error applying gate: {str(e)}")

def apply_cnot(state, control, target, num_qubits):
    try:
        dim = 2 ** num_qubits
        full_operator = np.eye(dim, dtype=complex)
        for i in range(dim):
            bin_i = format(i, f'0{num_qubits}b')
            if bin_i[control] == '1':
                target_bit = list(bin_i)
                target_bit[target] = '1' if bin_i[target] == '0' else '0'
                j = int("".join(target_bit), 2)
                full_operator[i, i], full_operator[i, j] = 0, 1
                full_operator[j, j], full_operator[j, i] = 0, 1
        return np.dot(full_operator, state)
    except Exception as e:
        raise ValueError(f"Error applying CNOT: {str(e)}")

def apply_toffoli(state, control1, control2, target, num_qubits):
    try:
        dim = 2 ** num_qubits
        full_operator = np.eye(dim, dtype=complex)
        for i in range(dim):
            bin_i = format(i, f'0{num_qubits}b')
            if bin_i[control1] == '1' and bin_i[control2] == '1':
                target_bit = list(bin_i)
                target_bit[target] = '1' if bin_i[target] == '0' else '0'
                j = int("".join(target_bit), 2)
                full_operator[i, i], full_operator[i, j] = 0, 1
                full_operator[j, j], full_operator[j, i] = 0, 1
        return np.dot(full_operator, state)
    except Exception as e:
        raise ValueError(f"Error applying Toffoli: {str(e)}")

def draw_probability_plot(final_state, num_qubits):
    # Visualization
    fig, ax = plt.subplots()
    ax.bar([f"|{bin(i)[2:].zfill(num_qubits)}\u27E9" for i in range(2**num_qubits)], np.abs(final_state) ** 2)
    ax.set_ylabel("Probability")
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    encoded_img = base64.b64encode(buf.getvalue()).decode('utf-8')
    return f'<img src="data:image/png;base64,{encoded_img}" width="400">'

def draw_circuit(instructions, num_qubits):
    max_width = 12
    num_gates = len(instructions.split("\n"))
    width = min(max_width, num_gates * 0.6)
    fig, ax = plt.subplots(figsize=(width, num_qubits * 0.8))
    ax.set_xlim(0, num_gates + 1)
    ax.set_ylim(-num_qubits, 1)
    ax.set_xticks([])
    ax.set_yticks(range(num_qubits))
    ax.set_yticklabels([f"q{i}" for i in range(num_qubits)])
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    for t, line in enumerate(instructions.split("\n")):
        parts = line.strip().split()
        if len(parts) >= 2:
            gate = parts[0].upper()
            qubits = [int(q[1:]) for q in parts[1:]]
            if gate == "CX":
                ax.plot([t + 1, t + 1], [qubits[0], qubits[1]], 'k', linewidth=1)
                ax.scatter(t + 1, qubits[0], s=100, color='black')  # Control qubit circle
                ax.text(t + 1, qubits[1], "CX", ha="center", va="center", fontsize=10, bbox=dict(facecolor='lightblue', edgecolor='black', boxstyle='round,pad=0.3'))
            elif gate == "CCX":
                ax.plot([t + 1, t + 1], [qubits[0], qubits[2]], 'k', linewidth=1)
                ax.scatter(t + 1, qubits[0], s=100, color='black')  # First control qubit
                ax.scatter(t + 1, qubits[1], s=100, color='black')  # Second control qubit
                ax.text(t + 1, qubits[2], "CCX", ha="center", va="center", fontsize=10, bbox=dict(facecolor='lightblue', edgecolor='black', boxstyle='round,pad=0.3'))
            else:
                for q in qubits:
                    ax.text(t + 1, q, gate, ha="center", va="center", fontsize=10, bbox=dict(facecolor='lightblue', edgecolor='black', boxstyle='round,pad=0.3'))
    
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    encoded_img = base64.b64encode(buf.getvalue()).decode('utf-8')
    return f'<div style="overflow-y:auto; max-height:300px;"><img src="data:image/png;base64,{encoded_img}" width="600"></div>'

def draw_circuit_1(instructions, num_qubits):
    max_width = 12
    num_gates = len(instructions.split("\n"))
    width = min(max_width, num_gates * 0.6)
    fig, ax = plt.subplots(figsize=(width, num_qubits * 0.8))
    ax.set_xlim(0, num_gates + 1)
    ax.set_ylim(-num_qubits, 1)
    ax.set_xticks([])
    ax.set_yticks(range(num_qubits))
    ax.set_yticklabels([f"q{i}" for i in range(num_qubits)])
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    for t, line in enumerate(instructions.split("\n")):
        parts = line.strip().split()
        if len(parts) >= 2:
            gate = parts[0].upper()
            qubits = [int(q[1:]) for q in parts[1:]]
            for q in qubits:
                ax.text(t + 1, q, gate, ha="center", va="center", fontsize=10, bbox=dict(facecolor='lightblue', edgecolor='black', boxstyle='round,pad=0.3'))
                if len(qubits) > 1:
                    ax.plot([t + 1, t + 1], [qubits[0], qubits[-1]], 'k', linewidth=1)
    
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    encoded_img = base64.b64encode(buf.getvalue()).decode('utf-8')
    return f'<div style="overflow-y:auto; max-height:300px;"><img src="data:image/png;base64,{encoded_img}" width="600"></div>'

def process_quantum_instructions(instructions: str, debug: bool = False):
    error_message = ""
    debug_info = ""
    try:
        lines = instructions.split("\n")
        num_qubits = max([int(parts[1][1:]) for line in lines if (parts := line.strip().split()) and len(parts) >= 2 and validate_qubit_format(parts[1])]) + 1
        initial_state = np.zeros(2**num_qubits, dtype=complex)
        initial_state[0] = 1
        final_state = initial_state

        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 2 and all(validate_qubit_format(q) for q in parts[1:]):
                gate = parts[0].lower()
                qubits = [int(q[1:]) for q in parts[1:]]
                if gate == "i":
                    final_state = apply_single_qubit_gate(identity(), final_state, qubits[0], num_qubits)
                elif gate == "h":
                    final_state = apply_single_qubit_gate(hadamard(), final_state, qubits[0], num_qubits)
                elif gate == "x":
                    final_state = apply_single_qubit_gate(pauli_x(), final_state, qubits[0], num_qubits)
                elif gate == "cx" and len(qubits) == 2:
                    final_state = apply_cnot(final_state, qubits[0], qubits[1], num_qubits)
                elif gate == "ccx" and len(qubits) == 3:
                    final_state = apply_toffoli(final_state, qubits[0], qubits[1], qubits[2], num_qubits)
                else:
                    error_message += f"Invalid instruction: {line}\n"
    
        img_html = draw_probability_plot(final_state, num_qubits)

        circuit_img = draw_circuit(instructions, num_qubits)

        return str(final_state), img_html, circuit_img, error_message, debug_info
    except Exception as e:
        return "", "", "", f"Error: {str(e)}", ""

iface = gr.Interface(
    fn=process_quantum_instructions,
    inputs=[
        gr.Textbox(lines=10, placeholder="Enter quantum instructions (e.g., H q0, CX q0 q1)"),
        gr.Checkbox(label="Enable Debug Mode")
    ],
    outputs=[
        gr.Textbox(label="Quantum State"),
        gr.HTML(label="State Visualization"),
        gr.HTML(label="Quantum Circuit"),
        gr.Textbox(label="Errors", interactive=False),
        gr.Textbox(label="Debug Info", interactive=False)
    ],
    title="Custom Quantum Processor Simulator",
    description="Enter instructions to manipulate a quantum circuit and visualize the state."
)

if __name__ == "__main__":
    iface.launch(debug=True)
