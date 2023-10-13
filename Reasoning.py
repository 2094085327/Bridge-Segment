import os
from datetime import datetime

import cv2
import gradio as gr
import numpy as np
from PIL import Image
from gradio import components
from skimage.morphology import skeletonize
from sklearn.neighbors import KDTree

import Config as Cf
import Logger as Log
import MaxCircle as Mc


def file_upload(_model_name, conf, img=None):
    """
    传递处理后图片
    Args:
        _model_name: 模型名称
        img: 待处理图片
        conf: 置信度

    Returns:
        new_images: 转移后的图片
    """
    global model, pro_img_list
    if _model_name == "":
        raise gr.Error("未选择模型")

    if img is None:
        raise gr.Error("未选择待处理图片")

    log.info(f"启用模型: {_model_name}")

    # 获取原始图片的宽度和高度
    height, width, _ = img.shape

    # 计算缩放后的宽度和高度
    if width * height > 512 * 512:
        img = cv2.resize(img, (int(width * 0.5), int(height * 0.5)))

    # 将图片转换为PIL图像对象
    img_pil = Image.fromarray(np.uint8(img))

    if not models_check:
        model = Cf.load_model(_model_name)

    load_path = os.path.join(models_path, _model_name)
    if load_path != model_path:
        model = Cf.load_model(_model_name, load_path)
        Cf.write_models(load_path, _model_name)

    else:
        Cf.write_models(os.path.join(models_path, _model_name), _model_name)

    # 进行推理
    model.predict(img_pil, save=True, save_txt=True, imgsz=640, conf=conf)

    new_images = Cf.copyFiles()
    log.info("推理完成")
    return new_images


def SVD(points):
    """
    使用 奇异值分解 （SVD）计算骨架线的法向量
    """
    # 二维，三维均适用
    # 二维直线，三维平面
    pts = points.copy()
    # 奇异值分解
    c = np.mean(pts, axis=0)
    A = pts - c  # shift the points
    A = A.T  # 3*n
    u, s, vh = np.linalg.svd(A, full_matrices=False, compute_uv=True)  # A=u*s*vh
    normal = u[:, -1]

    # 法向量归一化
    nlen = np.sqrt(np.dot(normal, normal))
    normal = normal / nlen
    # normal 是主方向的方向向量 与PCA最小特征值对应的特征向量是垂直关系
    # u 每一列是一个方向
    # s 是对应的特征值
    # c >>> 点的中心
    # normal >>> 拟合的方向向量
    return u, s, c, normal


def estimate_normals(points, n):
    """
    计算points表示的曲线上的每一个点法向量.
    等同于 estimate_normal_for_pos(points,points,n)

    Input：
    ------
    points: nx2 ndarray 曲线点集.
    n: 用到的近邻点的个数

    Output：
    ------
    normals: nx2 ndarray 在points曲线上的每一处的法向量.
    """

    pts = np.copy(points)
    tree = KDTree(pts, leaf_size=2)
    idx = tree.query(
        pts, k=n, return_distance=False, dualtree=False, breadth_first=False
    )
    # pts = np.concatenate((np.concatenate((pts[0].reshape(1,-1),pts),axis=0),pts[-1].reshape(1,-1)),axis=0)
    normals = []
    for i in range(0, pts.shape[0]):
        pts_for_normals = pts[idx[i, :], :]
        _, _, _, normal = SVD(pts_for_normals)
        normals.append(normal)
    normals = np.array(normals)
    return normals


def line_plane_intersection(line1, line2, point, normal):
    # 计算平面法向量和平面上一点
    plane_normal = normal / np.sqrt(np.dot(normal, normal))
    plane_point = point - np.dot(point, plane_normal) * plane_normal
    # 计算两条线段的方向向量
    line1_direction = line1 - line2
    line2_direction = line2 - line1

    # 计算平面法向量和两条线段的叉积
    cross_product = np.cross(line1_direction, plane_normal)
    # 计算交点
    t1 = np.dot(plane_point - line1, cross_product) / np.dot(
        line1_direction, cross_product
    )
    t2 = np.dot(plane_point - line2, cross_product) / np.dot(
        line2_direction, cross_product
    )
    intersection_point = line1 + t1 * line1_direction
    return intersection_point

def find_intersection(point, direction, contours_img):
    p1 = point - 1000 * direction
    p2 = point + 1000 * direction
    line = cv2.line(
        np.zeros_like(skeleton), tuple(p1.astype(int)), tuple(p2.astype(int)), 1, 1
    )
    intersections = cv2.findContours(
        np.asarray(line & contours_img), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if intersections:
        return intersections[0][0][0], intersections[-1][0][0]
    return None, None


def easy_mode2(inverted_img, org_img, width_threshold):
    # 查找轮廓
    contours_img = inverted_img.copy()
    contours, _ = cv2.findContours(
        np.asarray(contours_img), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # 创建一个全零的数组，与原图像具有相同的尺寸
    output_image = np.zeros_like(inverted_img)

    blobs = np.array(inverted_img)
    blobs = np.where(blobs == 255, 1, blobs)
    iw, ih = blobs.shape
    skeleton = skeletonize(blobs)
    x, y = np.where(skeleton > 0)

    skeleton_pixel = np.where(blobs is False, 0, 255)

    # skeleton_pixel = np.zeros((iw, ih, 3), dtype=np.uint8)
    # skeleton_pixel[skeleton, 1] = 255

    points = np.where(skeleton_pixel == 255)
    num_points = int(0.7 * len(points[0]))

    centers = np.hstack((x.reshape(-1, 1), y.reshape(-1, 1)))
    normals = estimate_normals(centers, 9)  # 这个用于估计法向量的KNN

    for i in range(centers.size):
        point = centers[i]
        normal = normals[i]
        # 计算与外轮廓相交的点
        intersection_point = None
        for contour in contours:
            contour = np.array(contour)
            if cv2.pointPolygonTest(contour, point, False) > 0:
                # 计算交点
                intersection_point = line_plane_intersection(
                    contour[0], contour[-1], point, normal
                )
                break
        if intersection_point is not None:
            # 绘制线段
            cv2.line(
                skeleton_pixel, tuple(point), tuple(intersection_point), (0, 255, 0), 2
            )

    # x_points = points[1][indices]
    # y_points = points[0][indices]
    # sorted_indices = np.argsort(x_points)
    # x_points = x_points[sorted_indices]
    # y_points = y_points[sorted_indices]
    #
    # # 对于每个点，计算一条垂线，使得它与骨架外轮廓相交
    # for i in range(num_points):
    #     x = x_points[i]
    #     y = y_points[i]
    #     # 计算垂线的斜率和截距
    #     slope, intercept = np.polyfit([x, x], [0, ih-1], 1)
    #     # 计算与骨架外轮廓相交的点
    #     y_intersect = int(intercept)
    #     x_intersect = int(x - intercept / slope)
    #     # 绘制垂线
    #     cv2.line(skeleton_pixel, (x, 0), (x, ih-1), (0, 255, 0), 1)
    #     # 绘制与骨架外轮廓相交的点
    #     cv2.circle(skeleton_pixel, (x_intersect, y_intersect), 3, (0, 0, 255), -1)
    #
    #
    # # 在全零图像上绘制轮廓，设置轮廓区域为1
    # cv2.drawContours(skeleton_pixel, contours, -1, (255,0,0))

    return skeleton_pixel


def get_finish_img(
    img, threshold, offset, noise_size, high_precision, easy_mode_open, width_threshold
):
    """
    获取计算裂缝宽度后的图片
    Args:
        img: 输入图片
        threshold: 阈值
        offset: 偏移量
        noise_size: 噪点阈值
        high_precision: 是否为高精度模式
        easy_mode_open: 是否为简易模式
        width_threshold: 宽度阈值

    Returns:
          返回处理后的图片
    """
    global finish_data, pro_img_list
    # 获取原始图片的宽度和高度
    height, width, _ = img.shape
    log.info(f"图片高度: {height}")
    log.info(f"图片宽度: {width}")
    # 缩放图片
    if width * height > 1920 * 1080:
        min_img = cv2.resize(img, (int(width * 0.5), int(height * 0.5)))
    else:
        min_img = img

    if len(pro_img_list) != 0:
        inverted_img = pro_img_list[4]
        log.info("检测到缓存，进行启用")
    else:
        inverted_img = process_images(
            img, noise_size, threshold, offset, easy_mode_open, width_threshold
        )[4]
        log.info("未检测到缓存，进行计算")

    log.info(f"二值化图片大小: {inverted_img.shape}")
    log.info(f"原图大小: {img.shape}")
    finish_img, all_data_list, skeleton_pixel = Mc.max_circle(
        inverted_img, min_img, high_precision
    )

    # 获取当前时间
    current_time = datetime.now().strftime("%Y%m%d%H%M%S")

    # 构建新的文件名
    new_file_name = f"{current_time}.jpg"

    cv2.imwrite(
        os.path.join(root_dir, "result", "inCircle", new_file_name),
        cv2.cvtColor(finish_img, cv2.COLOR_BGR2RGB),
    )

    finish_data = list(map(list, zip(*[all_data_list[0]])))
    if len(list(map(list, zip(*[all_data_list[1]])))) > 0:
        other_data = list(map(list, zip(*[all_data_list[1]])))
    else:
        other_data = [["无其余裂缝数据"]]

    width_data_list = [
        all_data_list[2][0],
        all_data_list[3][0],
        other_data,
        finish_data,
    ]
    Cf.width_json(current_time, width_data_list)

    random_data = update_page("初始化")

    pro_img_list = []

    return (
        [finish_img, skeleton_pixel],
        all_data_list[2][0],
        all_data_list[3][0],
        other_data,
        random_data,
    )


def process_images(
    pen_pro_img, noise_size, threshold, offset, easy_mode_open, width_threshold
):
    global pro_img_list

    # 获取原始图像的高度和宽度
    height, width, _ = pen_pro_img.shape

    # 将图像大小调整为原始大小的一半
    if width * height > 1920 * 1080:
        min_img = cv2.resize(pen_pro_img, (int(width * 0.5), int(height * 0.5)))
    else:
        min_img = pen_pro_img

    # 将图像转换为灰度
    gray_image = cv2.cvtColor(min_img, cv2.COLOR_BGR2GRAY)

    # 将高斯模糊应用于灰度图像
    kernel_size = 11  # 高斯核的大小
    sigma = 2  # 高斯核的标准差
    blurred_image = cv2.GaussianBlur(gray_image, (kernel_size, kernel_size), sigma)

    # 对模糊图像应用自适应阈值处理
    adaptive_threshold = cv2.adaptiveThreshold(
        blurred_image,
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

    # 将二值图像转换成反二值图像
    inverted_image = cv2.bitwise_not(binary_image)

    # 如果easy_mode_open为True，则执行其他处理
    if easy_mode_open:
        easy_image = easy_mode2(inverted_image, min_img, width_threshold)
        pro_img_list = [
            gray_image,
            blurred_image,
            adaptive_threshold,
            binary_image,
            inverted_image,
            easy_image,
        ]
    else:
        pro_img_list = [
            gray_image,
            blurred_image,
            adaptive_threshold,
            binary_image,
            inverted_image,
        ]

    return pro_img_list


def update_page(increase):
    """
    更新表格数据的函数
    Args:
        increase: 指令

    Returns:
        finish_data: 表格数据
    """
    global page
    total_pages = (len(finish_data) + 4) // 5
    if increase == "初始化":
        page = 0
    elif increase == "上一页" and page != 0:
        page -= 1
    elif increase == "下一页" and page < total_pages - 1:
        start = page * 5
        end = start + 5
        if len(finish_data[start:end]) < 5:
            return finish_data[start:end]
        page += 1

    start = page * 5
    end = start + 5
    return finish_data[start:end]


def reasoning(args, model_input):
    conf, threshold, offset, noise, wide, simple_line, high_precision = args
    with gr.Blocks() as reasoningDemo:
        # 垂直排列是默认情况，不加也没关系
        with gr.Row():
            img_input = components.Image(label="选择图片文件")
            gallery = gr.Gallery(label="推理结果", columns=1, rows=1, preview=True)

            filtered_image = gr.Gallery(label="高斯模糊", columns=1, rows=1, preview=True)

            wide_gallery = gr.Gallery(label="裂缝宽度结果", columns=1, rows=1, preview=True)
        with gr.Accordion(label="裂缝宽度计算结果", open=False):
            with gr.Row():
                max_wide_label = gr.Textbox(label="最大宽度", value="0")
                avg_wide_label = gr.Textbox(label="平均宽度", value="0")

            with gr.Row():
                second_label = gr.Dataframe(
                    headers=["次要裂缝最大宽度"],
                    datatype=["number"],
                    max_rows=5,
                    overflow_row_behaviour="paginate",
                )
                with gr.Column():
                    random_label = gr.Dataframe(headers=["随机取样宽度"], datatype=["number"])
                    with gr.Row():
                        before_page = "上一页"
                        next_page = "下一页"
                        before_data = gr.Button(before_page)
                        next_data = gr.Button(next_page)

        before_data.click(fn=lambda: update_page(before_page), outputs=[random_label])
        next_data.click(fn=lambda: update_page(next_page), outputs=[random_label])

        greet_btn = gr.Button("提交")

        greet_btn.click(
            fn=file_upload,
            inputs=[
                model_input,
                conf,
                img_input,
            ],
            outputs=[gallery],
        )

        greet_btn.click(
            fn=process_images,
            inputs=[img_input, noise, threshold, offset, simple_line, wide],
            outputs=[filtered_image],
        )

        greet_btn.click(
            fn=get_finish_img,
            inputs=[
                img_input,
                threshold,
                offset,
                noise,
                high_precision,
                simple_line,
                wide,
            ],
            outputs=[
                wide_gallery,
                max_wide_label,
                avg_wide_label,
                second_label,
                random_label,
            ],
        )

    return reasoningDemo


log = Log.HandleLog()
root_dir = Cf.root_dir
models_path = os.path.join(root_dir, "models")
models_check, model_path = Cf.check_models()
model_name = os.path.basename(model_path)
pro_img_list = []
if models_check:
    model = Cf.load_model(model_name, model_path)
