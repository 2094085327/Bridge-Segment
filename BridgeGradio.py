import os
import threading
import time
from datetime import datetime

import cv2
import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from gradio import components
from skimage import measure
from skimage.morphology import skeletonize
from sklearn.neighbors import KDTree
from ultralytics import YOLO

import Config as Cf
import Gallery as Gal
import ImageView as Iv
import IncircleWide as Iw
import MaxCircle as Mc


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


def calcu_dis_from_ctrlpts(ctrlpts):
    if ctrlpts.shape[1] == 4:
        return np.sqrt(np.sum((ctrlpts[:, 0:2] - ctrlpts[:, 2:4]) ** 2, axis=1))
    else:
        return np.sqrt(np.sum((ctrlpts[:, [0, 2]] - ctrlpts[:, [3, 5]]) ** 2, axis=1))


def estimate_normal_for_pos(pos, points, n):
    """
    计算pos处的法向量.

    Input：
    ------
    pos: nx2 ndarray 需要计算法向量的位置.
    points: 骨架线的点集
    n: 用到的近邻点的个数

    Output：
    ------
    normals: nx2 ndarray 在pos位置处的法向量.
    """

    # 估计给定点的法向量
    pts = np.copy(points)
    tree = KDTree(pts, leaf_size=2)
    idx = tree.query(
        pos, k=n, return_distance=False, dualtree=False, breadth_first=False
    )
    # pts = np.concatenate((np.concatenate((pts[0].reshape(1,-1),pts),axis=0),pts[-1].reshape(1,-1)),axis=0)
    normals = []
    for i in range(0, pos.shape[0]):
        pts_for_normals = pts[idx[i, :], :]
        _, _, _, normal = SVD(pts_for_normals)
        normals.append(normal)
    normals = np.array(normals)
    return normals


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


def get_crack_ctrlpts(centers, normals, bpoints, hband=5, vband=2, est_width=0):
    # 获得裂纹宽度的主要算法
    cpoints = np.copy(centers)
    cnormals = np.copy(normals)

    xmatrix = np.array([[0, 1], [-1, 0]])
    cnormalsx = np.dot(xmatrix, cnormals.T).T  # x轴的法线
    N = cpoints.shape[0]

    interp_segm = []
    widths = []
    for i in range(N):
        try:
            ny = cnormals[i]
            nx = cnormalsx[i]
            tform = np.array([nx, ny])
            bpoints_loc = np.dot(tform, bpoints.T).T
            cpoints_loc = np.dot(tform, cpoints.T).T
            ci = cpoints_loc[i]

            bl_ind = (bpoints_loc[:, 0] - (ci[0] - hband)) * (
                bpoints_loc[:, 0] - ci[0]
            ) < 0
            br_ind = (bpoints_loc[:, 0] - ci[0]) * (
                bpoints_loc[:, 0] - (ci[0] + hband)
            ) <= 0
            bl = bpoints_loc[bl_ind]  # 左侧边缘点
            br = bpoints_loc[br_ind]  # 右侧边缘点

            if est_width > 0:
                # 下面的数值 est_width 是预估计的裂缝宽度
                half_est_width = est_width / 2
                # 找到当前点的左侧边缘点
                # blt数组中所有点的第二个坐标值都小于当前点的第二个坐标值加上预估宽度的一半,其他同blt数组
                blt = bl[(bl[:, 1] - (ci[1] + half_est_width)) * (bl[:, 1] - ci[1]) < 0]
                blb = bl[(bl[:, 1] - (ci[1] - half_est_width)) * (bl[:, 1] - ci[1]) < 0]
                brt = br[(br[:, 1] - (ci[1] + half_est_width)) * (br[:, 1] - ci[1]) < 0]
                brb = br[(br[:, 1] - (ci[1] - half_est_width)) * (br[:, 1] - ci[1]) < 0]
            else:
                """
                当 est_width 为 0 时，代码会根据裂缝的左右边缘点，以及裂缝的中心点，计算出裂缝的宽度。
                根据裂缝的左右边缘点以及裂缝的中心点计算出裂缝的宽度。
                首先计算出左右边缘点的坐标，然后根据裂缝的中心点计算出左右边缘点与中心点的水平距离。
                根据预设的垂直范围 vband，判断左右边缘点是否在垂直范围内，如果在，则将其作为裂缝的边缘点；否则将其忽略。
                根据左右边缘点的坐标计算出裂缝的宽度，并将其存储在 widths 数组中。
                """
                blt = bl[bl[:, 1] > np.mean(bl[:, 1])]
                if np.ptp(blt[:, 1]) > vband:
                    blt = blt[blt[:, 1] > np.mean(blt[:, 1])]

                blb = bl[bl[:, 1] < np.mean(bl[:, 1])]
                if np.ptp(blb[:, 1]) > vband:
                    blb = blb[blb[:, 1] < np.mean(blb[:, 1])]

                brt = br[br[:, 1] > np.mean(br[:, 1])]
                if np.ptp(brt[:, 1]) > vband:
                    brt = brt[brt[:, 1] > np.mean(brt[:, 1])]

                brb = br[br[:, 1] < np.mean(br[:, 1])]
                if np.ptp(brb[:, 1]) > vband:
                    brb = brb[brb[:, 1] < np.mean(brb[:, 1])]

            if blt.size > 0:
                t1 = blt[np.argsort(blt[:, 0])[-1]]
                t2 = brt[np.argsort(brt[:, 0])[0]]

            else:
                # 如果数组为空，则设置t1为None
                t1 = None
                t2 = None

            # if blt.size == 0:
            #     interp1 = None
            #     interp2 = None
            # else:
            #     t1 = blt[np.argsort(blt[:, 0])[-1]]
            #     t2 = brt[np.argsort(brt[:, 0])[0]]
            #
            #     b1 = blb[np.argsort(blb[:, 0])[-1]]
            #     b2 = brb[np.argsort(brb[:, 0])[0]]
            #
            #     interp1 = (ci[0] - t1[0]) * ((t2[1] - t1[1]) / (t2[0] - t1[0])) + t1[1]
            #     interp2 = (ci[0] - b1[0]) * ((b2[1] - b1[1]) / (b2[0] - b1[0])) + b1[1]

            # t1 = blt[np.argsort(blt[:, 0])[-1]]
            # t2 = brt[np.argsort(brt[:, 0])[0]]

            b1 = blb[np.argsort(blb[:, 0])[-1]]
            b2 = brb[np.argsort(brb[:, 0])[0]]

            interp1 = (ci[0] - t1[0]) * ((t2[1] - t1[1]) / (t2[0] - t1[0])) + t1[1]
            interp2 = (ci[0] - b1[0]) * ((b2[1] - b1[1]) / (b2[0] - b1[0])) + b1[1]

            if interp1 - ci[1] > 0 and interp2 - ci[1] < 0:
                widths.append([i, interp1 - ci[1], interp2 - ci[1]])

                interps = np.array([[ci[0], interp1], [ci[0], interp2]])

                interps_rec = np.dot(np.linalg.inv(tform), interps.T).T

                interps_rec = interps_rec.reshape(1, -1)[0, :]
                interp_segm.append(interps_rec)
        except Exception as e:
            # print("the %d-th was wrong" % i)
            # traceback.print_exc()
            continue
    interp_segm = np.array(interp_segm)
    widths = np.array(widths)
    # check
    # show_2dpoints([np.array([[ci[0],interp1],[ci[0],interp2]]),np.array([t1,t2,b1,b2]),cpoints_loc,bl,br],[10,20,15,2,2])
    return interp_segm, widths


def load_model(load_model_name, load_model_path=None):
    """
    加载模型
    Args:
        load_model_name: 模型名字
        load_model_path: 模型路径

    Returns:
        model: 加载后的模型
    """
    global model
    if load_model_path is None and root_dir is not None:
        load_model_path = os.path.join(root_dir, "models", load_model_name)
        model = YOLO(load_model_path)
        print("模型加载完毕")
    elif load_model_path is not None:
        model = YOLO(load_model_path)

    return model


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


def auto_generate():
    """
    自动检测主进程
    Returns:
        None

    """
    global model, pro_img_list
    print("线程启动...")
    time.sleep(1)
    stop_event.clear()
    # 打开摄像头
    cap = cv2.VideoCapture(0)
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

        print("img: ", img.shape)
        print("总像素大小:", width * height)

        # 计算缩放后的宽度和高度
        if width * height > 1920 * 1080:
            img = cv2.resize(img, (int(width * 0.5), int(height * 0.5)))

        print("img_org: ", img_org.shape)
        # 将图片转换为PIL图像对象
        img_pil = Image.fromarray(np.uint8(img))

        if not models_check:
            model = load_model(model_name)

        load_path = os.path.join(models_path, model_name)
        if load_path != model_path:
            model = load_model(model_name, load_path)
            Cf.write_models(load_path, model_name)

        else:
            Cf.write_models(os.path.join(models_path, model_name), model_name)

        # 进行推理
        model.predict(img_pil, save=True, save_txt=True, imgsz=640, conf=auto_conf)

        Cf.copyFiles()

        pro_img_list = process_images(
            img, auto_noise, auto_threshold, auto_offset, auto_simple_line, auto_wide
        )
        print("预处理完成")

        get_finish_img(
            img_org,
            auto_threshold,
            auto_offset,
            auto_noise,
            auto_high_precision,
            auto_simple_line,
            auto_wide,
        )
        print("计算完成")
        time.sleep(9)

    print("线程已关闭")


def output_img():
    global pro_img_list
    while len(pro_img_list) == 0:
        time.sleep(1)
    # time.sleep(5)
    return pro_img_list


def thread_start():
    global thread
    thread = threading.Thread(target=auto_generate)
    thread.start()


def auto_start():
    print("开始自动检测")
    stop_event.clear()


def auto_stop():
    print("停止自动检测")
    stop_event.set()
    # thread.join()


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

    # 获取原始图片的宽度和高度
    height, width, _ = img.shape

    # 计算缩放后的宽度和高度
    if width * height > 512 * 512:
        img = cv2.resize(img, (int(width * 0.5), int(height * 0.5)))

    # 将图片转换为PIL图像对象
    img_pil = Image.fromarray(np.uint8(img))

    if not models_check:
        model = load_model(_model_name)

    load_path = os.path.join(models_path, _model_name)
    if load_path != model_path:
        model = load_model(_model_name, load_path)
        Cf.write_models(load_path, _model_name)

    else:
        Cf.write_models(os.path.join(models_path, _model_name), _model_name)

    # 进行推理
    model.predict(img_pil, save=True, save_txt=True, imgsz=640, conf=conf)

    new_images = Cf.copyFiles()
    return new_images


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


# 找出这条垂直线与轮廓的交点。
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


def easy_mode(inverted_img, width_threshold):
    # global pro_img_list
    # print(pro_img_list)
    # gray_image = pro_img_list[0]
    # blurred_image = pro_img_list[1]
    # binarization_img = pro_img_list[2]
    # conversion_img = pro_img_list[3]
    # inverted_img = pro_img_list[4]

    blobs = np.array(inverted_img)
    blobs = np.where(blobs == 255, 1, blobs)
    iw, ih = blobs.shape
    skeleton = skeletonize(blobs)

    x, y = np.where(skeleton > 0)
    centers = np.hstack((x.reshape(-1, 1), y.reshape(-1, 1)))

    normals = estimate_normals(centers, 9)  # 用于估计法向量的KNN

    # 搜索裂纹轮廓
    contours = measure.find_contours(blobs, 0.8)

    bl = contours[0]
    br = contours[1]

    bpoints = np.vstack((bl, br))

    bpixel = np.zeros((iw, ih, 3), dtype=np.uint8)
    bpoints = bpoints.astype(np.int64)
    bpixel[bpoints[:, 0], bpoints[:, 1], 0] = 255

    bpixel_and_skeleton = np.copy(bpixel)
    bpixel_and_skeleton[skeleton, 1] = 255

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))

    ax.imshow(bpixel_and_skeleton)

    interps, widths = get_crack_ctrlpts(
        centers, normals, bpoints, hband=2, vband=2, est_width=width_threshold
    )

    interps_show = interps[
        np.random.choice(interps.shape[0], 240, replace=True), :
    ]  # 由于太多，这里随机采样240个测量位置，进行显示

    for i in range(interps_show.shape[0]):
        ax.plot(
            [interps_show[i, 1], interps_show[i, 3]],
            [interps_show[i, 0], interps_show[i, 2]],
            c="c",
            ls="-",
            lw=1,
            marker="o",
            ms=2,
            mec="c",
            mfc="c",
        )

    fig.tight_layout()
    current_time = datetime.now().strftime("%Y%m%d%H%M%S")
    new_file_name = f"{current_time}.jpg"

    # 保存图片
    easy_path = os.path.join(root_dir, "result", "easyMode", new_file_name)
    plt.savefig(easy_path)

    # 读取保存的图片，并将其转换为ndarray类型
    output_image = cv2.imread(easy_path)

    return output_image


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
    print("height: ", height)
    print("width: ", width)
    # 缩放图片
    if width * height > 1920 * 1080:
        min_img = cv2.resize(img, (int(width * 0.5), int(height * 0.5)))
    else:
        min_img = img

    if len(pro_img_list) != 0:
        inverted_img = pro_img_list[4]
        print("使用缓存")
    else:
        inverted_img = process_images(
            img, noise_size, threshold, offset, easy_mode_open, width_threshold
        )[4]
        print("未使用缓存")

    print("img1: ", inverted_img.shape)
    print("img2: ", img.shape)
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


def get_page_data(img_list, page_num, items_per_page=20):
    """
    获取页面信息
    Args:
        img_list: 图片列表
        page_num: 页码
        items_per_page: 每页展示数量

    Returns:
        img_list: 需要显示的图片
    """
    start = page_num * items_per_page
    end = start + items_per_page
    return img_list[start:end]


if __name__ == "__main__":
    Cf.inspect_config_file()

    # 获取当前脚本文件的根目录
    root_dir = os.path.dirname(os.path.abspath(__file__))
    models_path = os.path.join(root_dir, "models")
    page = 0
    # pageNum = 0
    auto = False
    thread = None
    img_name = None
    model, gray_img_total = None, None
    finish_data = []
    # gallery_list = []
    pro_img_list = []
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

    models_check, model_path = Cf.check_models()
    model_name = os.path.basename(model_path)
    if models_check:
        load_model(model_name, model_path)
    with gr.Blocks() as demo:
        folder_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
        result_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "result")
        # 获取文件夹中的文件列表
        file_list = os.listdir(folder_path)

        # 初始化图片列表
        img_page_list = Gal.initialization_img_list("yolo")
        img_list = img_page_list[0:20]

        in_circle_list = Iw.initialization()

        # 创建一个文件组
        options = [
            file
            for file in file_list
            if os.path.isfile(os.path.join(folder_path, file))
        ]

        if models_check:
            model_input = components.Dropdown(
                choices=options, value=model_name, label="选择模型文件"
            )
        else:
            # 创建一个文件输入组件
            model_input = components.Dropdown(choices=options, label="选择模型文件")

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

        with gr.Tab("推理"):
            # Blocks特有组件，设置所有子组件按垂直排列
            # 垂直排列是默认情况，不加也没关系
            with gr.Row():
                img_input = components.Image(label="选择图片文件")
                gallery = gr.Gallery(label="推理结果", columns=1, rows=1, preview=True)

                filtered_image = gr.Gallery(
                    label="高斯模糊", columns=1, rows=1, preview=True
                )

                wide_gallery = gr.Gallery(
                    label="裂缝宽度结果", columns=1, rows=1, preview=True
                )
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
                        random_label = gr.Dataframe(
                            headers=["随机取样宽度"], datatype=["number"]
                        )
                        with gr.Row():
                            before_page = "上一页"
                            next_page = "下一页"
                            before_data = gr.Button(before_page)
                            next_data = gr.Button(next_page)

            before_data.click(
                fn=lambda: update_page(before_page), outputs=[random_label]
            )
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

        with gr.Tab("自监控模式"):
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
            stop_button.click(fn=auto_stop)
        with gr.Tab("图库浏览器"):
            # Blocks特有组件，设置所有子组件按水平排列
            Iv.imageView()

    config = Cf.read_config_file()
    port = config["database"]["port"]
    Cf.has_public_ip(port)
    demo.launch(server_name=config["database"]["host"], server_port=port)
