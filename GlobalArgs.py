import gradio as gr


def getAllArgs():
    """
    获取参数
    Args:
        get_conf:  置信度
        get_threshold:  阈值
        get_offset:  偏移量
        get_noise:  噪点大小
        get_high_precision:  高精度模式
        get_simple_line:  简单模式
        get_wide:

    Returns:
        None

    """
    print("conf:", get_conf)
    print("threshold:", get_threshold)
    print("offset:", get_offset)
    print("noise:", get_noise)
    print("high_precision:", get_high_precision)
    print("simple_line:", get_simple_line)
    print("wide:", get_wide)

    global auto_conf, auto_threshold, auto_offset, auto_noise, auto_high_precision, auto_simple_line, auto_wide

    auto_conf = get_conf
    auto_threshold = get_threshold
    auto_offset = get_offset
    auto_noise = get_noise
    auto_high_precision = get_high_precision
    auto_simple_line = get_simple_line
    auto_wide = get_wide



def allArgs():
    with gr.Blocks() as globalArgsDemo:

        with gr.Accordion(label="识别参数", open=False):
            conf = gr.Slider(label="置信度", minimum=0, maximum=1, value=0.6)
            threshold = gr.Slider(label="卷积核大小", minimum=0, maximum=255, value=161)
            offset = gr.Slider(label="偏移值大小", minimum=0, maximum=100, value=31)
            noise = gr.Slider(label="噪点过滤阈值", minimum=0, maximum=500, value=300)
            wide = gr.Slider(label="宽度计算阈值", minimum=0, maximum=500, value=60)

        with gr.Accordion(label="检测模式", open=False):
            # auto_camera = gr.Checkbox(label="自动监测模式", value=False)
            simple_line = gr.Checkbox(label="单裂缝模式", value=False)
            high_precision = gr.Checkbox(label="高精度模式(大幅增加计算时间)", value=False)

    args=(conf, threshold, offset, noise, wide, simple_line, high_precision)

    return globalArgsDemo,args