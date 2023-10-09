import os

import gradio as gr

import Config as Cf
import Gallery as Gal

pageNum = 0
page = 0
img_name = ""
img_page_list = []
gallery_list = []
finish_data = []


def initialization():
    """
    初始化图片列表
    Returns:
        img_list: 图片列表

    """
    global img_page_list
    # 初始化图片列表
    img_page_list = Gal.initialization_img_list("inCircle")
    img_list = img_page_list[0:20]
    return img_list


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

    return img_page_list[start:end], pageNum, delete_wide.update(visible=False)


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


def on_select_img(gallery_list_now, event_data: gr.SelectData):
    """
    被选择的图片
    Args:
        gallery_list_now: 图片列表
        event_data: 当前选中的图片

    Returns:
          delete 按钮状态
    """
    global img_name, gallery_list, finish_data, page
    page = 0
    gallery_list = gallery_list_now
    img_name = os.path.basename(gallery_list_now[event_data.index]["name"])

    width_data = Cf.read_width_json(img_name.split(".")[0])

    finish_data = width_data["random_width"]

    return (
        delete_wide.update(visible=True),
        width_data["max_width"],
        width_data["avg_width"],
        width_data["other_width"],
        finish_data[0:5],
    )


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

    os.remove(os.path.join(file_path, "inCircle", img_name_delete))
    os.remove(
        os.path.join(file_path, "inCircle", img_name_delete.split(".")[0] + ".json")
    )
    return img_page_list[start:end]


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
        img_page_list = Gal.initialization_img_list("inCircle")
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

    return page_data, pageNum, delete_wide.update(visible=False)


def update_page(increase):
    """
    更新表格数据
    Args:
        increase: 更新表格数据的具体操作

    Returns:
        finish_data: 更新后的表格数据

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


def inCircleWide():
    """
    宽度计算
    Returns:
        circleWideDemo: 宽度计算界面组件

    """
    global delete_wide
    result_path = Cf.result_path

    with gr.Blocks() as circleWideDemo:
        with gr.Row():
            with gr.Column():
                with gr.Row():
                    beforeButton_wide = gr.Button("上一页")
                    nextButton_wide = gr.Button("下一页")
                    getPageNum_wide = gr.Number(label="页码", interactive=True)
                    refresh_wide = gr.Button("🔄")
                    first_page_wide = gr.Button("首页")
                    end_page_wide = gr.Button("尾页")
                    in_circle_list = initialization()
                with gr.Column():
                    wide_result_img = gr.Gallery(
                        label="宽度计算结果",
                        value=in_circle_list,
                        columns=5,
                        object_fit="contain",
                    )
                    delete_wide = gr.Button("删除", visible=False)

            with gr.Column():
                max_width = gr.Textbox(label="最大宽度")
                avg_width = gr.Textbox(label="平均宽度")
                other_width = gr.Dataframe(
                    headers=["次要裂缝宽度"],
                    datatype=["number"],
                    max_rows=5,
                    overflow_row_behaviour="paginate",
                )

                random_width = gr.Dataframe(headers=["随机取样宽度"], datatype=["number"])

                data_previous = gr.Button("上一页")
                data_next = gr.Button("下一页")

        first_page_wide.click(
            fn=lambda: update_img_list("首页"),
            outputs=[wide_result_img, getPageNum_wide, delete_wide],
        )

        beforeButton_wide.click(
            fn=lambda: update_img_list("上一页"),
            outputs=[wide_result_img, getPageNum_wide, delete_wide],
        )

        nextButton_wide.click(
            fn=lambda: update_img_list("下一页"),
            outputs=[wide_result_img, getPageNum_wide, delete_wide],
        )

        end_page_wide.click(
            fn=lambda: update_img_list("尾页"),
            outputs=[wide_result_img, getPageNum_wide, delete_wide],
        )

        refresh_wide.click(
            fn=lambda: update_img_list("刷新"),
            outputs=[wide_result_img, getPageNum_wide, delete_wide],
        )

        getPageNum_wide.submit(
            fn=on_page_num_change,
            inputs=[getPageNum_wide],
            outputs=[wide_result_img, getPageNum_wide, delete_wide],
        )

        wide_result_img.select(
            fn=on_select_img,
            inputs=[wide_result_img],
            outputs=[delete_wide, max_width, avg_width, other_width, random_width],
        )

        delete_wide.click(
            fn=lambda: delete_img(img_name, result_path), outputs=[wide_result_img]
        )

        data_next.click(fn=lambda: update_page("下一页"), outputs=[random_width])
        data_previous.click(fn=lambda: update_page("上一页"), outputs=[random_width])
    return circleWideDemo


delete_wide = gr.Button("删除", visible=False)
