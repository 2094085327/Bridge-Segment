import os
import threading
import time

import cv2
import gradio as gr
import numpy as np
from PIL import Image

import Config as Cf
import Logger as Log
import Reasoning as Re


def get_args(
    get_conf,
    get_threshold,
    get_offset,
    get_noise,
    get_high_precision,
    get_simple_line,
    get_wide,
):
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
    log.info(f"置信度conf: {get_conf}")
    log.info(f"阈值threshold: {get_threshold}")
    log.info(f"偏移量offset: {get_offset}")
    log.info(f"噪点大小noise: {get_noise}")
    log.info(f"高精度模式high_precision: {get_high_precision}")
    log.info(f"简单模式simple_line: {get_simple_line}")
    log.info(f"宽度计算阈值wide: {get_wide}")

    global auto_conf, auto_threshold, auto_offset, auto_noise, auto_high_precision, auto_simple_line, auto_wide

    auto_conf = get_conf
    auto_threshold = get_threshold
    auto_offset = get_offset
    auto_noise = get_noise
    auto_high_precision = get_high_precision
    auto_simple_line = get_simple_line
    auto_wide = get_wide


def thread_start():
    """
    启动自动检测
    Returns:
        None
    """
    global thread
    thread = threading.Thread(target=auto_generate)
    thread.start()


def thread_stop():
    """
    停止自动检测
    Returns:
        None
    """
    log.info("停止自动检测")
    stop_event.set()


def auto_generate():
    """
    自动检测主进程
    Returns:
        None

    """
    global model, pro_img_list
    log.info("线程启动...")
    stop_event.clear()
    # 打开摄像头
    cap = cv2.VideoCapture(0)
    time.sleep(1)
    # 调整图像大小
    width = 800
    height = 600
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    while not stop_event.is_set():
        hx, img = cap.read()
        img = np.array(img)

        # 获取原始图片的宽度和高度
        height, width, _ = img.shape
        img_org = img.copy()

        log.info(f"图片大小: {img.shape}")
        log.info(f"总像素大小: {width * height}")

        # 计算缩放后的宽度和高度
        if width * height > 1920 * 1080:
            img = cv2.resize(img, (int(width * 0.5), int(height * 0.5)))
            log.info(f"缩放后图片大小: {img.shape}")

        # 将图片转换为PIL图像对象
        img_pil = Image.fromarray(np.uint8(img))

        if not models_check:
            model = Cf.load_model(model_name)

        load_path = os.path.join(models_path, model_name)
        if load_path != model_path:
            model = Cf.load_model(model_name, load_path)
            Cf.write_models(load_path, model_name)

        else:
            Cf.write_models(os.path.join(models_path, model_name), model_name)

        # 进行推理
        model.predict(img_pil, save=True, save_txt=True, imgsz=640, conf=auto_conf)

        Cf.copyFiles()

        pro_img_list = Re.process_images(
            img, auto_noise, auto_threshold, auto_offset, auto_simple_line, auto_wide
        )
        log.info("预处理完成")

        Re.get_finish_img(
            img_org,
            auto_threshold,
            auto_offset,
            auto_noise,
            auto_high_precision,
            auto_simple_line,
            auto_wide,
        )
        log.info("裂缝计算完成")
        time.sleep(9)

    log.warning("线程已关闭")


def output_img():
    """
    输出图片
    Returns:
        pro_img_list: 输出图片列表
    """
    global pro_img_list
    while len(pro_img_list) == 0:
        time.sleep(0.1)
    return pro_img_list


def autoCheck(args):
    """
    自动检测模式模组
    Args:
        args: 参数

    Returns:
        autoCheckDemo: 自动检测模式模组
    """
    conf, threshold, offset, noise, wide, simple_line, high_precision = args
    with gr.Blocks() as autoCheckDemo:
        with gr.Row():
            run_button = gr.Button("启动")
            stop_button = gr.Button("停止")
        with gr.Row():
            out_img = gr.Gallery(label="预处理图片", columns=1, rows=1, preview=True)

        run_button.click(
            fn=get_args,
            inputs=[
                conf,
                threshold,
                offset,
                noise,
                high_precision,
                simple_line,
                wide,
            ],
        )
        run_button.click(fn=thread_start)
        run_button.click(fn=output_img, outputs=[out_img])
        stop_button.click(fn=thread_stop)

    return autoCheckDemo


log = Log.HandleLog()
root_dir = Cf.root_dir
models_path = os.path.join(root_dir, "models")
models_check, model_path = Cf.check_models()
model_name = os.path.basename(model_path)
pro_img_list = []
thread = None
if models_check:
    model = Cf.load_model(model_name, model_path)

stop_event = threading.Event()
stop_event.set()
# 自动监测模式全局变量
(
    auto_conf,
    auto_threshold,
    auto_offset,
    auto_noise,
    auto_high_precision,
    auto_simple_line,
    auto_wide,
) = (0.6, 161, 31, 300, False, False, 60)
