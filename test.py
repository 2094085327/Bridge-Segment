import gradio as gr


def calculator(num1):
    return num1


demo = gr.Interface(
    calculator,
    ["number"],
    "number",
    live=True,
)
demo.launch()
