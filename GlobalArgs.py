import gradio as gr


def allArgs():
    """
    全局参数

    Returns:
        globalArgsDemo:  全局参数
    """
    with gr.Blocks() as globalArgsDemo:
        with gr.Accordion(label="识别参数", open=False):
            conf = gr.Slider(label="置信度", minimum=0, maximum=1, value=0.6)
            threshold = gr.Slider(label="卷积核大小", minimum=0, maximum=255, value=161)
            offset = gr.Slider(label="偏移值大小", minimum=0, maximum=100, value=31)
            noise = gr.Slider(label="噪点过滤阈值", minimum=0, maximum=500, value=300)
            wide = gr.Slider(label="宽度计算阈值", minimum=0, maximum=500, value=60)

        with gr.Accordion(label="检测模式", open=False):
            simple_line = gr.Checkbox(label="单裂缝模式", value=False)
            high_precision = gr.Checkbox(label="高精度模式(大幅增加计算时间)", value=False)

    args = (conf, threshold, offset, noise, wide, simple_line, high_precision)

    return globalArgsDemo, args
