import math
import random

import cv2
import numpy as np
from scipy import ndimage
from skimage.morphology import skeletonize


def get_min_max_coordinates(c):
    left_x = min(c[:, 0, 0])
    right_x = max(c[:, 0, 0])
    down_y = max(c[:, 0, 1])
    up_y = min(c[:, 0, 1])
    return left_x, right_x, up_y, down_y


def get_contour_points(c, left_x, right_x, up_y, down_y, Nx, Ny):
    pixel_X = np.linspace(left_x, right_x, Nx)
    pixel_Y = np.linspace(up_y, down_y, Ny)
    xx, yy = np.meshgrid(pixel_X, pixel_Y)
    in_list = [
        (xx[i][j], yy[i][j])
        for i in range(Nx)
        for j in range(Ny)
        if cv2.pointPolygonTest(c, (xx[i][j], yy[i][j]), False) > 0
    ]
    return np.array(in_list)


def is_overlap(center1, radius1, center2, radius2):
    """
    判断两圆是否相交或者相离
    Args:
        center1: 圆1的圆心
        radius1: 圆1的半径
        center2: 圆2的圆心
        radius2: 圆2的半径
    Returns:
        True: 有重合或包含关系
        False: 相离
    """
    # 计算两圆的坐标
    x1, y1 = center1
    x2, y2 = center2
    # 计算两圆的距离
    distance = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
    # 判断两圆是否有重叠或者包含的关系
    if radius1 + radius2 <= distance:
        # 两圆相离
        return False
    else:
        # 两圆内切内含相交
        return True


def get_each_max(in_point, distance_transform):
    # 初始化最大半径和对应的圆心坐标
    max_radius = 0
    circle_center = (0, 0)
    for in_point_list in in_point.tolist():
        center = tuple(in_point_list)
        min_distance = distance_transform[center[0], center[1]]
        # 更新最大半径和对应的圆心坐标
        if min_distance > max_radius:
            max_radius = min_distance
            circle_center = (center[1], center[0])

    return max_radius, circle_center


def random_loop(target_length, in_point, random_circle_list, distance_transform):
    i = 0
    while len(random_circle_list) < target_length:
        # print(in_point)
        if i > 10000:
            print("运行循环超出10000次，跳出循环")
            break
        i = i + 1
        selected_points = in_point[np.random.choice(len(in_point), 1, replace=False)]

        center = tuple(selected_points.tolist()[0])
        center = (center[1], center[0])
        # cv2 距离变换检测
        max_radius = distance_transform[center[1], center[0]]
        if len(random_circle_list) == 0:
            random_circle_list.append([max_radius, center])

        else:
            overlap_found = False
            for circle in random_circle_list:
                before_radius, before_center = circle
                if is_overlap(center, max_radius, before_center, before_radius):
                    overlap_found = True
                    break

            if not overlap_found:  # 如果没有发现重叠
                random_circle_list.append([max_radius, center])

    return random_circle_list


def max_circle(binarization_image, img_original, high_precision):
    """
    计算轮廓内切圆算法
    Args:
        binarization_image: 输入图片，图片需为二值化图像
        img_original: 原始图片
        high_precision: 是否采用高精度计算
    Returns:
        img_original: 绘制好宽度的图片
    """
    # 寻找二值图像的轮廓
    contous, _ = cv2.findContours(
        np.asarray(binarization_image), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )

    print("binarization_image: ", np.asarray(binarization_image).shape)

    print("len(contous): ", len(contous))
    if len(contous) == 0:
        return img_original, [[None], [None], [None], [None]], []

        # 创建一个全零的数组，与原图像具有相同的尺寸
    output_image = np.zeros_like(binarization_image)

    # 在全零图像上绘制轮廓，设置轮廓区域为1
    cv2.drawContours(output_image, contous, -1, 1, thickness=cv2.FILLED)

    expansion_circle_list = []  # 所有裂缝最大内切圆半径和圆心列表
    random_circle_list = []  # 随机选择的裂缝内切圆列表
    # 可能一张图片中存在多条裂缝，对每一条裂缝进行循环计算
    max_len = max(len(c) for c in contous)
    finally_average_width_list = []

    # TODO 当其他裂缝过多时只取5条，先运算最大裂缝
    binarization_image_np = np.asarray(binarization_image)
    height, width = binarization_image_np.shape[:2]
    skeleton = skeletonize(binarization_image_np)

    skeleton_pixel = np.where(binarization_image_np == False, 0, 255)


    # 创建8连通结构元素
    structure = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
    cracks = []

    # 连通组件分析
    labeled, num_features = ndimage.label(skeleton.astype(int), structure=structure)
    # 创建一个列表来存储每一条裂缝的骨架
    for i in range(1, num_features + 1):
        # 获取单独的裂缝骨架
        crack_skeleton = np.where(labeled == i, 1, 0)

        # 将其添加到列表中
        cracks.append(crack_skeleton)

    print("添加完成")
    # 计算距离变换
    distance_transform = cv2.distanceTransform(
        output_image.astype(np.uint8), cv2.DIST_L2, 3
    )

    for c in contous:
        get_crack = None
        for crack in cracks:
            # 找到所有值为1的点的坐标
            y, x = np.where(crack == 1)
            # 如果有多个点，从中随机选择一个
            if len(x) > 0 and len(y) > 0:
                index = random.randint(0, len(x) - 1)
                random_x, random_y = x[index], y[index]

                if (
                    cv2.pointPolygonTest(
                        c, (float(random_x), float(random_y)), measureDist=False
                    )
                    >= 0
                ):
                    get_crack = crack
                    break
        in_point = np.argwhere(get_crack == 1)

        radius, center = get_each_max(in_point, distance_transform)

        expansion_circle_list.append([radius, center])  # 保存每条裂缝最大内切圆的半径和圆心
        if radius != 0:
            random_circle_list.append([radius, center])

        average_width_list = []

        if len(c) == max_len:
            if high_precision:
                random_circle_list.sort(key=lambda x: x[0], reverse=True)

                TARGET_LENGTHS = [10] * 9 + [20]
            else:
                TARGET_LENGTHS = [20]

            for target_length in TARGET_LENGTHS:
                random_circle_list = random_loop(
                    target_length, in_point, random_circle_list, distance_transform
                )
                average_width = sum(circle[0] for circle in random_circle_list) / len(
                    random_circle_list
                )
                average_width_list.append(round(average_width * 2, 2))

                # 如果不是最后一次迭代，则截断列表
                if target_length == 10:
                    random_circle_list = random_circle_list[:1]

            finally_average_width_list.append(
                round(sum(average_width_list) / len(average_width_list), 2)
            )
        else:
            continue

    print("平均长度计算完成")
    random_wide_list = [round(circle[0] * 2, 2) for circle in random_circle_list]
    expansion_circle_radius_list = [
        circle[0] for circle in expansion_circle_list
    ]  # 每条裂缝最大内切圆半径列表
    max_radius = max(expansion_circle_radius_list)
    max_center = expansion_circle_list[expansion_circle_radius_list.index(max_radius)][
        1
    ]
    max_wide_list = [round(max_radius * 2, 2)]
    # 从random_circle_list中移除第一个
    random_circle_list.pop(0)

    # 绘制轮廓
    cv2.drawContours(img_original, contous, -1, (0, 0, 255), -1)

    secondary_wide_list = []
    # 绘制随机内切圆
    for random_circle in random_circle_list:
        radius_s, center_s = random_circle
        cv2.circle(
            img_original,
            (int(center_s[0]), int(center_s[1])),
            int(radius_s),
            (255, 255, 0),
            2,
        )

    # 绘制裂缝轮廓最大内切圆
    for expansion_circle in expansion_circle_list:
        radius_s, center_s = expansion_circle
        if radius_s == max_radius:  # 最大内切圆，用红色标注
            cv2.circle(
                img_original,
                (int(max_center[0]), int(max_center[1])),
                int(max_radius),
                (255, 0, 0),
                2,
            )

        else:  # 其他内切圆，用青色标注
            secondary_wide_list.append(round(radius_s * 2, 2))
            if radius_s != 0:
                cv2.circle(
                    img_original,
                    (int(center_s[0]), int(center_s[1])),
                    int(radius_s),
                    (0, 255, 255),
                    2,
                )

    all_data_list = [
        random_wide_list,
        secondary_wide_list,
        max_wide_list,
        finally_average_width_list,
    ]
    return img_original, all_data_list, skeleton_pixel
