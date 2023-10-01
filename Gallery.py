import os


def initialization_img_list(img_dir):
    result_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'result',img_dir)

    img_page_list = []
    for root, dirs, files in os.walk(result_path):
        for file in files:
            if file.endswith(".jpg") or file.endswith(".png"):
                img_path = os.path.join(root, file)
                img_page_list.append(img_path)
    return img_page_list
