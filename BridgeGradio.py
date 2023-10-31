import os
from datetime import datetime

import cv2
import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
from gradio import components
from skimage import measure
from skimage.morphology import skeletonize
from sklearn.neighbors import KDTree

import AutoCheck as Ac
import Config as Cf
import GlobalArgs as Ga
import ImageView as Iv
import IncircleWide as Iw
import Logger as Log
import Reasoning as Re
import Calculate as Ca


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
            log.error("计算裂缝宽度时出现错误")
            log.error(e)
            continue
    interp_segm = np.array(interp_segm)
    widths = np.array(widths)
    # check
    # show_2dpoints([np.array([[ci[0],interp1],[ci[0],interp2]]),np.array([t1,t2,b1,b2]),cpoints_loc,bl,br],[10,20,15,2,2])
    return interp_segm, widths


def easy_mode(inverted_img, width_threshold):
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


if __name__ == "__main__":
    # 初始化配置文件
    Cf.inspect_config_file()

    # 获取当前脚本文件的根目录
    root_dir = Cf.root_dir

    models_check, model_path = Cf.check_models()
    model_name = os.path.basename(model_path)

    folder_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
    result_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "result")

    # 获取文件夹中的文件列表
    file_list = os.listdir(folder_path)

    # 初始化图片列表
    in_circle_list = Iw.initialization()

    # 创建一个文件组
    options = [
        file for file in file_list if os.path.isfile(os.path.join(folder_path, file))
    ]

    # 创建一个文件输入组件
    # Blocks特有组件，设置所有子组件按垂直排列
    with gr.Blocks() as demo:
        if models_check:
            model_input = components.Dropdown(
                choices=options, value=model_name, label="选择模型文件"
            )
        else:
            model_input = components.Dropdown(choices=options, label="选择模型文件")

        _, args = Ga.allArgs()

        # Blocks特有组件，设置所有子组件按水平排列
        with gr.Tab("相机标定"):
            # 相机标定组件
            Ca.calculate(args)

        with gr.Tab("推理"):
            # 推理组件
            Re.reasoning(args, model_input)

        with gr.Tab("自监控模式"):
            # 自监控组件
            Ac.autoCheck(args)

        with gr.Tab("图库浏览器"):
            # 图库组件
            Iv.imageView()

        with gr.Tab("设置"):
            # 宽度计算组件
            Cf.globalConfig(model_input)

    Cf.system_check()
    config = Cf.read_config_file()
    port = config["database"]["port"]
    log_rank = config["logging"]["level"]
    Cf.has_public_ip(port)
    log = Log.HandleLog()
    log.info(f"启用的日志等级: {log_rank}")
    Cf.remove_cache()
    try:
        demo.launch(server_name=config["database"]["host"], server_port=port)

    except Exception as e:
        log.error("启动失败,请检查端口是否被占用")
