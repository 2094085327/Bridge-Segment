import errno
import json
import os
import socket
import re
import shutil
from datetime import datetime

from PIL import Image
from ultralytics import YOLO

root_dir = os.path.dirname(os.path.abspath(__file__))
result_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "result")


def inspect_config_file():
    """
    检查当前目录下是否存在config.json文件，如果不存在则创建文件
    """
    if not os.path.exists("config.json"):
        config = {
            "database": {
                "host": "0.0.0.0",
                "port": 7860,
                "username": "root",
                "password": "password",
            },
            "logging": {"level": "info", "file": "app.log"},
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
        with open(file_path, "rb+") as f:
            return False
    except IOError as e:
        if e.errno == errno.EACCES:
            return True  # 文件被占用
        elif e.errno == errno.ENOENT:
            return False  # 文件不存在
        else:
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
                print(f"文件 {file_path} 正在被其他程序占用，无法复制。")
        except Exception as e:
            print(f"无法打开图像文件 {file_path}: {str(e)}")

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
    # global model
    if load_model_path is None and root_dir is not None:
        load_model_path = os.path.join(root_dir, "models", load_model_name)
        model = YOLO(load_model_path)
        print("模型加载完毕")
    elif load_model_path is not None:
        model = YOLO(load_model_path)

    return model