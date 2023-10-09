import os

import gradio as gr

import Config as Cf
import Gallery as Gal
import IncircleWide as Iw

pageNum = 0
img_name = None
gallery_list = []
img_page_list = Gal.initialization_img_list("yolo")
delete = gr.Button("删除", visible=False)


def on_select_img(gallery_list_now, event_data: gr.SelectData):
    """
    被选择的图片
    Args:
        gallery_list_now: 图片列表
        event_data: 当前选中的图片

    Returns:
          delete 按钮状态
    """
    global img_name, gallery_list
    gallery_list = gallery_list_now
    filename = os.path.basename(gallery_list_now[event_data.index]["name"])
    img_name = filename

    return delete.update(visible=True)


def get_page_boundaries(page_num, items_per_page, total_items):
    """
    获取当前页的起始和结束位置
    Args:
        page_num: 当前页码
        items_per_page: 每页显示的图片数量
        total_items: 图片总数

    Returns:
        start: 起始位置
        end: 结束位置
        page_num: 当前页码
    """
    start = page_num * items_per_page
    end = start + items_per_page

    if start >= total_items:
        page_num = (total_items - 1) // items_per_page
        start = page_num * items_per_page
        end = start + items_per_page

    return start, end, page_num


def on_page_num_change(event, items_per_page=20):
    """
    更新当前页码
    Args:
        event: 当前页码
        items_per_page: 每页显示的图片数量

    Returns:
        img_page_list: 更新后的图片列表
        page_num: 当前页码
        delete: 删除按钮是否可见
    """
    global pageNum
    pageNum = max(int(event), 0)  # 保证pageNum非负

    start, end, pageNum = get_page_boundaries(
        pageNum, items_per_page, len(img_page_list)
    )

    return img_page_list[start:end], pageNum, delete.update(visible=False)


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


def update_img_list(increase):
    """
    更新图片列表
    Args:
        increase: 更新图片的方法
    Returns:
        page_data: 更新后的图片列表
        page_num: 当前页码
        delete: 删除按钮的状态
    """
    global img_page_list, pageNum

    total_pages = (len(img_page_list) + 19) // 20  # 计算总页数

    def init_page():
        return 0

    def last_page():
        return total_pages - 1

    def current_page():
        return pageNum

    def prev_page():
        return max(0, pageNum - 1)

    def next_gallery_page():
        if pageNum < total_pages - 1:
            return pageNum + 1
        return pageNum

    def refresh_page():
        global img_page_list
        img_page_list = Gal.initialization_img_list("yolo")
        return pageNum

    actions = {
        "初始化": init_page,
        "首页": init_page,
        "刷新": refresh_page,
        "尾页": last_page,
        "跳转": current_page,
        "上一页": prev_page,
        "下一页": next_gallery_page,
    }

    pageNum = actions[increase]()
    page_data = get_page_data(img_page_list, pageNum)

    return page_data, pageNum, delete.update(visible=False)


def delete_img(img_name_delete, file_path):
    """
    删除图片
    Args:
        img_name_delete: 图片名称
        file_path: 图片路径

    Returns:
        img_page_list: 删除后的图片列表
    """
    global img_page_list

    # 使用列表推导来过滤掉目标字符串
    filtered_file_paths = [
        path for path in img_page_list if not path.endswith(img_name_delete)
    ]

    start = pageNum * 20
    end = start + 20
    img_page_list = filtered_file_paths

    os.remove(os.path.join(file_path, "yolo", img_name_delete))
    return img_page_list[start:end]


def imageView():
    """
    图片展示组件
    Returns:
        imageViewDemo: 图片展示组件

    """
    global delete, img_page_list
    result_path = Cf.result_path
    # 初始化图片列表
    img_list = img_page_list[0:20]
    with gr.Blocks() as imageViewDemo:
        with gr.Row():
            with gr.Tab("识别结果"):
                with gr.Row():
                    first_page = gr.Button("首页")
                    beforeButton = gr.Button("上一页")
                    getPageNum = gr.Number(label="页码", interactive=True)
                    refresh = gr.Button("🔄")
                    nextButton = gr.Button(
                        "下一页",
                    )
                    end_page = gr.Button("尾页")
                inference_results = gr.Gallery(
                    label="推理结果", value=img_list, columns=5, object_fit="contain"
                )
                delete = gr.Button("删除", visible=False)
            with gr.Tab("宽度计算"):
                with gr.Row():
                    block2 = Iw.inCircleWide()

        first_page.click(
            fn=lambda: update_img_list("首页"),
            outputs=[inference_results, getPageNum, delete],
        )
        beforeButton.click(
            fn=lambda: update_img_list("上一页"),
            outputs=[inference_results, getPageNum, delete],
        )
        nextButton.click(
            fn=lambda: update_img_list("下一页"),
            outputs=[inference_results, getPageNum, delete],
        )
        end_page.click(
            fn=lambda: update_img_list("尾页"),
            outputs=[inference_results, getPageNum, delete],
        )
        refresh.click(
            fn=lambda: update_img_list("刷新"),
            outputs=[inference_results, getPageNum, delete],
        )
        getPageNum.submit(
            fn=on_page_num_change,
            inputs=[getPageNum],
            outputs=[inference_results, getPageNum, delete],
        )

        inference_results.select(
            fn=on_select_img, inputs=[inference_results], outputs=[delete]
        )
        delete.click(
            fn=lambda: delete_img(img_name, result_path),
            outputs=[inference_results],
        )

    return imageViewDemo
