import json
import os


def inspect_config_file():
    """
    检查当前目录下是否存在config.json文件，如果不存在则创建文件
    """
    if not os.path.exists("config.json"):
        config = {
            "database": {
                "host": "localhost",
                "port": 3306,
                "username": "root",
                "password": "password"
            },
            "logging": {
                "level": "info",
                "file": "app.log"
            }
        }
        with open("config.json", "w") as f:
            f.write(json.dumps(config))


def write_models(path, name):
    with open("config.json", "r") as f:
        config = json.load(f)
    config["models"] = {
        "path": path,
        "name": name
    }
    with open("config.json", "w") as f:
        f.write(json.dumps(config))


def check_models():
    with open("config.json", "r") as f:
        config = json.load(f)
    if "models" in config:
        return True, config["models"]["path"]
    else:
        return False, ""


def width_json(json_name, data_list):
    json_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'result', 'inCircle', f'{json_name}.json')
    width_data = {
        "max_width": data_list[0],
        "avg_width": data_list[1],
        "other_width": data_list[2],
        "random_width": data_list[3],
    }

    with open(json_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(width_data, ensure_ascii=False))


def read_width_json(json_name):
    json_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'result', 'inCircle', f'{json_name}.json')
    with open(json_path, "r", encoding="utf-8") as f:
        width_data = json.load(f)
    return width_data
