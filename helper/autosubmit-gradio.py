import gradio as gr

def echo(text):
    return text  # Returns the same text to show in output

custom_js = """
function handleKeyDown(event) {
    if (event.key === 'Enter') {
        event.preventDefault();  // Prevent default behavior (which is submit)
        let textbox = event.target;
        let cursorPos = textbox.selectionStart;
        
        // Insert a newline at the cursor position
        let textBefore = textbox.value.substring(0, cursorPos);
        let textAfter = textbox.value.substring(cursorPos);
        textbox.value = textBefore + "\\n" + textAfter;
        textbox.selectionStart = textbox.selectionEnd = cursorPos + 1;

        // Manually trigger the Gradio submit event
        textbox.dispatchEvent(new Event('input', { bubbles: true }));
    }
}

function attachEventListener() {
    let inputBox = document.querySelector('textarea'); // Select the textbox
    if (inputBox) {
        inputBox.addEventListener('keydown', handleKeyDown);
    }
}

document.addEventListener("DOMContentLoaded", attachEventListener);
"""

with gr.Blocks() as demo:
    gr.HTML(f"<script>{custom_js}</script>")  # Inject JavaScript
    input_box = gr.Textbox(label="Input", placeholder="Type something...", lines=3)
    output_box = gr.Textbox(label="Output")
    
    input_box.submit(echo, inputs=input_box, outputs=output_box)  # Auto-submit on Enter

demo.launch()
