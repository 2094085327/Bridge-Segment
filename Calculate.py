import asyncio
import threading
import time

import cv2
import gradio as gr
import numpy as np

import Config as Cf
import Logger as Log

pro_img_list = []
args_get = (None, None, None, None, None, None, None)
num_args = (None, None, None)
thread = None
pre = 1
log = Log.HandleLog()
stop_event = threading.Event()
stop_event.set()
img_output_camera = gr.Gallery()


def count(img):
    global pre
    log.info(f"图片大小为：{img.shape}")

    contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours2 = sorted(contours, key=lambda a: cv2.contourArea(a), reverse=True)
    contours2 = contours2[1:]
    max_area = max(cv2.contourArea(c) for c in contours2)
    calculate_config = Cf.read_config_file()["calculate"]
    proportion = max(calculate_config["length"], calculate_config["width"]) / min(
        calculate_config["length"], calculate_config["width"]
    )
    for c in contours2:
        area = cv2.contourArea(c)
        box = cv2.minAreaRect(c)
        width = box[1][0]
        height = box[1][1]

        if min(width, height) == 0:
            continue
        log.info(f"长宽：{width,height}")
        log.info(f"长宽比例为：{max(width, height) / min(width, height)}")

        box = cv2.boxPoints(box)

        box = np.array(box, dtype="int")
        cv2.drawContours(img, [box], 0, (255, 255, 255), 1)
        cv2.imwrite("contours.png", img)

        get_proportion = max(width, height) / min(width, height)
        if area == max_area:
            if abs(get_proportion - proportion) > 50:
                log.warning("当前测得长宽比例与实际参照物比例差别较大")
                log.warning(f"测得长宽比例为：{get_proportion}")
                log.warning(f"实际长宽比例为：{proportion}")
                log.warning("请尝试重新校准摄像头或测量参照物长宽以减小误差")
            else:
                log.info(f"测得长宽比例为：{get_proportion}")
                log.info(f"实际长宽比例为：{proportion}")
            pre = max(width, height) / max(
                calculate_config["length"], calculate_config["width"]
            )
            log.info(f"测得 {pre} px/mm")

        else:
            continue
    return pre


def process_images(pen_pro_img, noise_size, threshold, offset):
    """
    图像预处理
    Args:
        pen_pro_img: 待处理图片
        noise_size: 噪点大小
        threshold: 阈值
        offset: 偏移量

    Returns:
        pro_img_list: 处理后的图片列表
    """
    global pro_img_list

    # 获取原始图像的高度和宽度
    height, width, _ = pen_pro_img.shape
    log.info(f"原始图像的高度为: {height}, 宽度为: {width}")

    # 将图像转换为灰度
    gray_image = cv2.cvtColor(pen_pro_img, cv2.COLOR_BGR2GRAY)
    # 对模糊图像应用自适应阈值处理
    adaptive_threshold = cv2.adaptiveThreshold(
        gray_image,
        255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY,
        threshold,
        offset,
    )

    # 将阈值图像转换为二值化图像
    binary_image = adaptive_threshold.copy()

    # 噪点过滤(将小于一定大小的黑色区域转换为白色)
    contours, hierarchy = cv2.findContours(
        binary_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )
    for cnt in contours:
        if cv2.contourArea(cnt) < noise_size:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(binary_image, (x, y), (x + w, y + h), (255, 255, 255), -1)

    pro_img_list = [
        pen_pro_img,
        gray_image,
        adaptive_threshold,
        binary_image,
    ]

    return pro_img_list


def thread_start(noise, threshold, offset):
    """
    启动自动检测
    Returns:
        None
    """
    global thread, num_args
    # noise, threshold, offset = num_args
    thread = threading.Thread(
        target=camera_process_images, args=(noise, threshold, offset)
    )
    thread.start()


def thread_stop():
    """
    停止自动检测
    Returns:
        None
    """
    log.info("停止自动检测")
    stop_event.set()


def camera_process_images(noise_size, threshold, offset):
    global pro_img_list, img_output_camera
    log.info("线程启动")
    stop_event.clear()
    log.warning(noise_size)
    log.warning(threshold)
    log.warning(offset)
    while not stop_event.is_set():
        cap = cv2.VideoCapture(0)

        time.sleep(1)
        # 调整图像大小
        width = 800
        height = 600
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        hx, img = cap.read()
        cap.release()
        img = np.array(img)

        # 获取原始图片的宽度和高度
        height, width, _ = img.shape

        log.info(f"原始图像的高度为: {height}, 宽度为: {width}")

        # 将图像转换为灰度
        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # 对模糊图像应用自适应阈值处理
        adaptive_threshold = cv2.adaptiveThreshold(
            gray_image,
            255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY,
            threshold,
            offset,
        )

        # 将阈值图像转换为二值化图像
        binary_image = adaptive_threshold.copy()

        # 噪点过滤(将小于一定大小的黑色区域转换为白色)
        contours, hierarchy = cv2.findContours(
            binary_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )

        for cnt in contours:
            if cv2.contourArea(cnt) < noise_size:
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(binary_image, (x, y), (x + w, y + h), (255, 255, 255), -1)

        pro_img_list = [
            img,
            gray_image,
            adaptive_threshold,
            binary_image,
        ]

        img_output_camera.update(value=[pro_img_list])
        # return gr.Gallery(label="预览图片",value=[pro_img_list])
    log.warning("线程已退出")


def view_img():
    global pro_img_list
    return pro_img_list


# 定义一个异步函数，用于运行Python程序并返回输出
async def run_process_images(img, noise, threshold, offset):
    loop = asyncio.get_running_loop()
    future = loop.run_in_executor(None, process_images, img, noise, threshold, offset)
    output = await future
    return output


def get_args(noise, threshold, offset):
    global num_args
    log.warning(noise)
    log.warning(threshold)
    log.warning(offset)
    num_args = (noise, threshold, offset)


def calculate(args):
    global args_get, img_output_camera
    args_get = args
    conf, threshold, offset, noise, wide, simple_line, high_precision = args
    with gr.Blocks() as reasoningDemo:
        with gr.Tab("图像模式"):
            view = gr.Button(value="预览")
            start = gr.Button(value="开始标定")
            with gr.Row():
                img_input = gr.Image(label="选择图片")
                img_output = gr.Gallery(label="预览图片")

        with gr.Tab("摄像机模式"):
            with gr.Row():
                view_camera = gr.Button(value="预览")
                stop_camera = gr.Button(value="停止预览")
                read_img = gr.Button(value="读取图片")
                start_camera = gr.Button(value="开始标定")
            with gr.Row():
                # img_input = gr.Image(label="选择图片")
                img_output_camera = gr.Gallery(label="预览图片")

    view.click(
        fn=process_images,
        inputs=[img_input, noise, threshold, offset],
        outputs=[img_output],
    )
    start.click(fn=lambda: count(pro_img_list[3]))

    # view_camera.click(get_args, inputs=[noise, threshold, offset])
    view_camera.click(thread_start, inputs=[noise, threshold, offset])
    read_img.click(fn=lambda: view_img(), outputs=[img_output_camera])
    stop_camera.click(thread_stop)
    start_camera.click(fn=lambda: count(pro_img_list[3]))

    return reasoningDemo
