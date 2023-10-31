import errno
import glob
import json
import os
import re
import shutil
import socket
import subprocess
import sys
from datetime import datetime, timedelta

import gradio as gr
from PIL import Image
from ultralytics import YOLO

import Logger as Log

root_dir = os.path.dirname(os.path.abspath(__file__))
result_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "result")
models_path = os.path.join(root_dir, "models")
log = Log.HandleLog()


def inspect_config_file():
    """
    检查当前目录下是否存在config.json文件，如果不存在则创建文件
    """
    if not os.path.exists("config.json"):
        log.warning("未检测到config.json文件，正在创建...")
        config = {
            "database": {
                "host": "0.0.0.0",
                "port": 7860,
                "username": "root",
                "password": "password",
            },
            "logging": {"level": "INFO", "file": "app.log"},
            "cache": {
                "open": False,
                "path": "cache",
                "days": 7,
                "inCircle": False,
                "yolo": False,
                "log": False,
            },
            "calculate": {"length": 0, "width": 0},
        }
        with open("config.json", "w") as f:
            f.write(json.dumps(config))


def read_config_file():
    """
    读取config.json文件
    """
    with open("config.json", "r") as f:
        config = json.load(f)
    return config


def is_private_ip(ip):
    """
    判断IP地址是否为私网IP
    Args:
        ip: IP地址

    Returns:
        返回检测状态

    """
    # 私网IP地址范围
    private_ranges = [
        ("10.0.0.0", "10.255.255.255"),
        ("172.16.0.0", "172.31.255.255"),
        ("192.168.0.0", "192.168.255.255"),
    ]

    # 将IP地址转换为整数
    ip_int = (
        int(ip.split(".")[0]) << 24
        | int(ip.split(".")[1]) << 16
        | int(ip.split(".")[2]) << 8
        | int(ip.split(".")[3])
    )

    # 判断IP地址是否在私网IP地址范围内
    for start, end in private_ranges:
        start_int = (
            int(start.split(".")[0]) << 24
            | int(start.split(".")[1]) << 16
            | int(start.split(".")[2]) << 8
            | int(start.split(".")[3])
        )
        end_int = (
            int(end.split(".")[0]) << 24
            | int(end.split(".")[1]) << 16
            | int(end.split(".")[2]) << 8
            | int(end.split(".")[3])
        )
        if start_int <= ip_int <= end_int:
            return False
    print("IP网段位于公网网段内，进行启用...")
    return True


def has_public_ip(port):
    """
    检索公网IP
    Args:
        port: 端口号

    Returns:
        返回检测状态

    """
    address = socket.getaddrinfo(socket.gethostname(), None)
    print("检索到本机IP，提供访问地址")
    pattern = r"(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})"
    for addr in address:
        match = re.search(pattern, addr[4][0])
        if match:
            if is_private_ip(match.group(1)):
                print("检索到公网IP，进行启用...")
                print(f"Running on public URL:  http://{match.group(1)}:{port}")
            else:
                print(f"http://{match.group(1)}:{port}")
    return False


def write_models(path, name):
    """
    写入模型路径
    Args:
        path: 模型路径
        name: 模型名称

    Returns:
        None
    """
    with open("config.json", "r") as f:
        config = json.load(f)
    config["models"] = {"path": path, "name": name}
    with open("config.json", "w") as f:
        f.write(json.dumps(config))


def check_models():
    """
    检查模型路径是否存在
    Returns:
        返回检测状态和模型路径
    """
    with open("config.json", "r") as f:
        config = json.load(f)
    if "models" in config:
        return True, config["models"]["path"]
    else:
        return False, ""


def width_json(json_name, data_list):
    """
    写入宽度数据
    Args:
        json_name: json文件名称
        data_list: 宽度数据列表

    Returns:
        None

    """
    json_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "result",
        "inCircle",
        f"{json_name}.json",
    )
    width_data = {
        "max_width": data_list[0],
        "avg_width": data_list[1],
        "other_width": data_list[2],
        "random_width": data_list[3],
    }

    with open(json_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(width_data, ensure_ascii=False))


def write_area_json():
    # 指定文件夹路径
    folder_path = os.path.join(root_dir, "result", "inCircle")

    # 查找以txt为后缀的文件，并按照修改时间排序
    txt_files = sorted(
        glob.glob(os.path.join(folder_path, "*.json")),
        key=os.path.getmtime,
        reverse=True,
    )

    # 获取最新的文件名
    if txt_files:
        latest_file = txt_files[0]
        latest_file_name = os.path.basename(latest_file)

        json_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "result",
            "inCircle",
            f"{latest_file_name}",
        )

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        area_cache_path = os.path.join(root_dir, "cache", "cache_area.json")
        if os.path.exists(area_cache_path):
            with open(area_cache_path, "r", encoding="utf-8") as f:
                area_cache = json.load(f)

            data["proportion"] = area_cache["proportion"]
            with open(json_path, "w", encoding="utf-8") as f:
                f.write(json.dumps(data, ensure_ascii=False))

            os.remove(area_cache_path)

        cache_path = os.path.join(root_dir, "cache", "cache.json")
        if os.path.exists(cache_path):
            with open(cache_path, "r", encoding="utf-8") as f:
                cache = json.load(f)

            data["crack_type"] = cache["crack_type"]

            with open(json_path, "w", encoding="utf-8") as f:
                f.write(json.dumps(data, ensure_ascii=False))

            os.remove(cache_path)
    else:
        log.error("未检测到txt文件，请检查文件夹")
        return


def read_width_json(json_name):
    """
    读取宽度数据
    Args:
        json_name:  json文件名称

    Returns:
        width_data: 返回宽度数据

    """
    json_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "result",
        "inCircle",
        f"{json_name}.json",
    )
    with open(json_path, "r", encoding="utf-8") as f:
        width_data = json.load(f)
    return width_data


def is_file_in_use(file_path):
    """
    检查文件是否被其他程序占用
    Args:
        file_path: 文件路径
    Returns:
        True: 文件被占用
        False: 文件未被占用
    """
    if not os.path.exists(file_path):
        return False  # 文件不存在，因此未被占用

    try:
        with open(file_path, "rb+"):
            return False
    except IOError as e:
        if e.errno == errno.EACCES:
            log.error(f"文件 {file_path} 正在被其他程序占用，无法复制。")
            return True  # 文件被占用
        elif e.errno == errno.ENOENT:
            log.error(f"文件 {file_path} 不存在，无法复制。")
            return False  # 文件不存在
        else:
            log.error(f"文件 {file_path} 未知错误，无法复制。")
            return True  # 其他未知错误，假设文件被占用


def copyFiles():
    """
    将Yolo处理后生成的图片进行转移，放置到对应文件夹中
    Args:

    Returns:
        new_img: 转移后的图片
    """
    directory = os.path.join(root_dir, "runs/segment")
    folders = [
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if os.path.isdir(os.path.join(directory, f))
    ]
    folders.sort(key=os.path.getctime)

    newest_folder_path = folders[-1] if folders else None
    if not newest_folder_path:
        return []

    image_files = (
        os.path.join(root, file)
        for root, _, files in os.walk(newest_folder_path)
        for file in files
        if file.endswith((".jpg", ".jpeg", ".png", ".gif"))
    )
    txt_path = os.path.join(
        root_dir, "runs", "segment", "predict", "labels", "image0.txt"
    )
    item_type = None
    if os.path.exists(txt_path):
        with open(txt_path, "r") as f:
            first_char = f.read(1)
            match first_char:
                case "0":
                    item_type = "竖状裂缝"
                case "1":
                    item_type = "块状裂缝"
                case "2":
                    item_type = "网状裂缝"
                case "3":
                    item_type = "横状裂缝"
        log.info(f"检测到裂缝类型为: {item_type}")

    else:
        log.warning("未检测到裂缝类型，无法进行分类")

    cache_path = os.path.join(root_dir, "cache", "cache.json")
    cache_file = {"crack_type": item_type}
    # 判断文件是否存在
    if not os.path.exists(cache_path):
        # 如果文件不存在，则创建文件夹并创建文件
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(cache_file, ensure_ascii=False))

    new_images = []
    for file_path in image_files:
        try:
            current_time = datetime.now().strftime("%Y%m%d%H%M%S")
            new_file_name = f"{current_time}.jpg"
            destination_path = os.path.join(root_dir, "result", "yolo", new_file_name)

            if not is_file_in_use(file_path):
                shutil.copy(file_path, destination_path)
                image = Image.open(destination_path)
                new_images.append(image)
            else:
                log.warning(f"文件 {file_path} 正在被其他程序占用，无法复制。")
        except Exception as e:
            log.error(f"无法打开图像文件 {file_path}: {str(e)}")

    # 删除所有文件夹
    for folder in folders:
        shutil.rmtree(folder)

    return new_images


def load_model(load_model_name, load_model_path=None):
    """
    加载模型
    Args:
        load_model_name: 模型名字
        load_model_path: 模型路径

    Returns:
        model: 加载后的模型
    """
    if load_model_path is None and root_dir is not None:
        load_model_path = os.path.join(root_dir, "models", load_model_name)
        model = YOLO(load_model_path, task="segment")
        log.info("模型加载完毕")
    elif load_model_path is not None:
        model = YOLO(load_model_path, task="segment")

    return model


def check_log_info(log_rank):
    """
    检查日志等级
    Args:
        log_rank: 日志等级

    Returns:
        返回日志等级
    """
    match log_rank:
        case "调试":
            return "DEBUG"
        case "信息":
            return "INFO"
        case "警告":
            return "WARNING"
        case "错误":
            return "ERROR"
        case "严重错误":
            return "CRITICAL"


def check_log_rank(log_rank):
    """
    转换日志等级
    Args:
        log_rank: 日志等级

    Returns:
        返回日志等级
    """
    match log_rank:
        case "DEBUG":
            return "调试"
        case "INFO":
            return "信息"
        case "WARNING":
            return "警告"
        case "ERROR":
            return "错误"
        case "CRITICAL":
            return "严重错误"


def save_config(
    cache_clean,
    _length,
    _width,
    _log_info,
    _model_name,
    circle_clean,
    yolo_clean,
    log_clean,
    cache_time,
):
    """
    保存配置
    Args:
        cache_clean: 是否自动清理缓存
        _length: 参照物长度
        _width: 参照物宽度
        _log_info: 日志等级
        _model_name: 模型名称
        circle_clean: 是否清理宽度计算结果
        yolo_clean: 是否清理识别结果
        log_clean: 是否清理日志文件
        cache_time: 缓存天数

    Returns:
        None
    """
    with open("config.json", "r") as f:
        config = json.load(f)
    config["cache"]["open"] = cache_clean
    config["cache"]["inCircle"] = circle_clean
    config["cache"]["yolo"] = yolo_clean
    config["cache"]["log"] = log_clean
    config["cache"]["days"] = cache_time
    config["calculate"]["length"] = _length
    config["calculate"]["width"] = _width
    config["logging"]["level"] = check_log_info(_log_info)
    load_path = os.path.join(models_path, _model_name)
    config["models"] = {"path": load_path, "name": _model_name}

    try:
        with open("config.json", "w") as f:
            f.write(json.dumps(config))
            log.info("配置保存成功")
            # gr.Info("配置保存成功")

    except BaseException as e:
        log.error(f"配置保存失败: {str(e)}")
        # raise gr.Warning("出现错误，请检查日志")


def system_check():
    """
    检查系统类型
    Returns:
        返回使用的重启文件类型
    """
    if sys.platform.startswith("win"):
        log.info("检测到系统为Windows")
        return "restart.bat"
    elif sys.platform.startswith("linux"):
        log.info("检测到系统为Linux")
        return "restart.sh"
    elif sys.platform.startswith("darwin"):
        log.info("检测到系统为MacOS")
        return "restart.sh"
    else:
        log.error("未检测到系统类型，请检查系统环境")


def restart_server():
    """
    重启服务
    Returns:
        None
    """
    log.warning("正在重启服务...")
    restart_file = system_check()
    bat_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), restart_file)
    log.warning(bat_path)

    subprocess.call(bat_path, shell=True)


def remove_file(file_path, days_ago):
    """
    删除文件
    Args:
        file_path: 文件路径
        days_ago: 天数

    Returns:

    """
    file_path = file_path
    for filename in os.listdir(file_path):
        full_file_path = os.path.join(file_path, filename)  # 创建新的变量来存储完整的文件路径

        # 获取文件的修改时间
        modification_time = datetime.fromtimestamp(os.path.getmtime(full_file_path))

        # 如果文件的修改时间早于设置天数前，则删除文件
        if modification_time < days_ago:
            try:
                os.remove(full_file_path)

            except OSError as e:
                log.error(f"删除文件失败: {str(e)}")
                continue


def remove_cache():
    """
    清理缓存文件
    Returns:
        None
    """
    config = read_config_file()
    # 设置配置的时间戳
    days_ago = datetime.now() - timedelta(days=config["cache"]["days"])

    if config["cache"]["open"]:
        if config["cache"]["inCircle"]:
            # 遍历指定文件夹下的所有文件
            log.info("正在清理宽度计算结果文件...")
            folder_path = os.path.join(result_path, "inCircle")
            remove_file(folder_path, days_ago)
            log.info("宽度计算结果清理完毕")
        if config["cache"]["yolo"]:
            log.info("正在清理识别结果...")
            folder_path = os.path.join(result_path, "yolo")
            remove_file(folder_path, days_ago)
            log.info("识别结果清理完毕")

        if config["cache"]["log"]:
            log.info("正在清理日志缓存...")
            folder_path = os.path.join(root_dir, "cache")
            remove_file(folder_path, days_ago)
            log.info("日志缓存清理完毕")

    log.info("缓存文件清理完毕")


def globalConfig(args):
    """
    全局配置
    Returns:
        configDemo: 配置界面
    """
    config = read_config_file()
    model_input = args
    with gr.Blocks() as configDemo:
        with gr.Row():
            save = gr.Button("保存配置")
            restart = gr.Button("重启服务")

        with gr.Column():
            with gr.Row():
                cache_clean = gr.Checkbox(
                    label="自动清理缓存文件", value=config["cache"]["open"]
                )
                inCircle_clean = gr.Checkbox(
                    label="清理宽度计算结果", value=config["cache"]["inCircle"]
                )
                yolo_clean = gr.Checkbox(label="清理识别结果", value=config["cache"]["yolo"])
                log_clean = gr.Checkbox(label="清理日志文件", value=config["cache"]["log"])
            cache_time = gr.Number(
                label="缓存天数", interactive=True, value=config["cache"]["days"]
            )

            log_info = gr.Dropdown(
                ["调试", "信息", "警告", "错误", "严重错误"],
                label="日志等级",
                value=check_log_rank(config["logging"]["level"]),
            )
            with gr.Row():
                gr.Text(show_label=False, value="参照物长度", interactive=False)
                length = gr.Number(
                    show_label=False,
                    interactive=True,
                    value=config["calculate"]["length"],
                )
                gr.Text(show_label=False, value="mm", interactive=False)

            with gr.Row():
                gr.Text(show_label=False, value="参照物宽度", interactive=False)
                width = gr.Number(
                    show_label=False,
                    interactive=True,
                    value=config["calculate"]["width"],
                )
                gr.Text(show_label=False, value="mm", interactive=False)

    save.click(
        fn=save_config,
        inputs=[
            cache_clean,
            length,
            width,
            log_info,
            model_input,
            inCircle_clean,
            yolo_clean,
            log_clean,
            cache_time,
        ],
    )
    restart.click(fn=restart_server)
    return configDemo
