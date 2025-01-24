import gradio as gr
import numpy as np

# Available quantum gates
gates = ["I", "X", "H", "Z"]  # Identity, Pauli-X, Hadamard, Pauli-Z

def initialize_grid(n_qubits, n_steps):
    """Initialize a blank grid with identity operations."""
    return [["I" for _ in range(n_steps)] for _ in range(n_qubits)]

def update_grid(grid, n_qubits, n_steps):
    """Ensure the grid size updates dynamically when steps or qubits are added/removed."""
    while len(grid) < n_qubits:
        grid.append(["I"] * len(grid[0]))
    while len(grid[0]) < n_steps:
        for row in grid:
            row.append("I")
    return [row[:n_steps] for row in grid[:n_qubits]]

def apply_gates(grid):
    """Simulate quantum state based on selected gates, starting from |0> state."""
    n_qubits = len(grid)
    state = np.zeros((2**n_qubits,), dtype=complex)
    state[0] = 1  # Initial state |0>
    
    for step in range(len(grid[0])):
        for q in range(n_qubits):
            gate = grid[q][step]
            if gate == "X":
                for i in range(2**n_qubits):
                    if (i >> q) & 1 == 0:
                        i_swap = i | (1 << q)
                        state[i], state[i_swap] = state[i_swap], state[i]
            elif gate == "H":
                for i in range(2**n_qubits):
                    if (i >> q) & 1 == 0:
                        i_other = i | (1 << q)
                        temp = state[i]
                        state[i] = (state[i] + state[i_other]) / np.sqrt(2)
                        state[i_other] = (temp - state[i_other]) / np.sqrt(2)
            elif gate == "Z":
                for i in range(2**n_qubits):
                    if (i >> q) & 1 == 1:
                        state[i] *= -1
    
    return list(np.round(state.real, 3))  # Return real part of final state

def interactive_circuit():
    n_qubits = gr.State(2)
    n_steps = gr.State(3)
    grid = gr.State(initialize_grid(n_qubits.value, n_steps.value))
    
    with gr.Blocks() as demo:
        with gr.Row():
            add_qubit = gr.Button("+ Qubit")
            remove_qubit = gr.Button("- Qubit")
            add_step = gr.Button("+ Step")
            remove_step = gr.Button("- Step")
        
        output_state = gr.Textbox("", label="Qubit State")

        grid_inputs = []
        for i in range(n_qubits.value):
            with gr.Row():
                row_inputs = []
                for j in range(n_steps.value):
                    dropdown = gr.Dropdown(gates, value="I", interactive=True)
                    dropdown.change(lambda val, i=i, j=j: update_grid_selection(val, i, j), inputs=[dropdown], outputs=[output_state])
                    row_inputs.append(dropdown)
                grid_inputs.append(row_inputs)

        def update_grid_selection(value, i, j):
            grid.value[i][j] = value
            return  apply_gates(grid.value)
        

        
        def update_buttons(action):
            if action == "add_qubit":
                n_qubits.value += 1
            elif action == "remove_qubit" and n_qubits.value > 1:
                n_qubits.value -= 1
            elif action == "add_step":
                n_steps.value += 1
            elif action == "remove_step" and n_steps.value > 1:
                n_steps.value -= 1
            
            grid.value = update_grid(grid.value, n_qubits.value, n_steps.value)
            grid.value = apply_gates(grid.value)
            return grid.value
        
        # def update_grid_values():
        #     updated_grid = [[dropdown.value for dropdown in row] for row in grid_inputs]
        #     return apply_gates(updated_grid)
        
        # for row in grid_inputs:
        #     for dropdown in row:
        #         dropdown.change(update_grid_values, inputs=[dropdown], outputs=[output_state])

        add_qubit.click(lambda: update_buttons("add_qubit"), outputs=[output_state])
        remove_qubit.click(lambda: update_buttons("remove_qubit"), outputs=[output_state])
        add_step.click(lambda: update_buttons("add_step"), outputs=[output_state])
        remove_step.click(lambda: update_buttons("remove_step"), outputs=[output_state])

    return demo

demo = interactive_circuit()
demo.launch()
