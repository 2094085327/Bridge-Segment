import os

def extract_timestamp(s):
    """
    提取时间戳
    Args:
        s: 文件名

    Returns:
        时间戳
    """
    return int(s.split('.')[0])

def sort_img_list(img_page_list):
    """
    对图片列表进行排序
    Args:
        img_page_list: 图片列表

    Returns:
        排序后的图片列表
    """
    sorted_list = sorted(img_page_list, key=lambda x: extract_timestamp(os.path.basename(x)))
    return sorted_list


def initialization_img_list(img_dir):
    """
    初始化图片列表
    Args:
        img_dir: 图片文件夹

    Returns:
        img_page_list: 图片列表
    """
    result_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'result',img_dir)

    img_page_list = []
    for root, dirs, files in os.walk(result_path):
        for file in files:
            if file.endswith(".jpg") or file.endswith(".png"):
                img_path = os.path.join(root, file)
                img_page_list.append(img_path)

    img_page_list = sort_img_list(img_page_list)
    return img_page_list
