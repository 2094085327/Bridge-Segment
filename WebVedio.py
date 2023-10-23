import re
import socket

import cv2
from flask import Flask, render_template, Response

app = Flask(__name__)
app.template_folder = "./"  # 指定模板文件的路径


# 视频采集函数
def video_stream():
    camera = cv2.VideoCapture(0)

    width = 800
    height = 600
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    print("启动摄像头")
    while True:
        ret, frame = camera.read()
        if not ret:
            break
        else:
            # 将帧转换为JPEG格式
            ret, jpeg = cv2.imencode(".jpg", frame)
            frame = jpeg.tobytes()
            yield (
                b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n\r\n"
            )


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    return Response(
        video_stream(), mimetype="multipart/x-mixed-replace; boundary=frame"
    )


if __name__ == "__main__":
    address = socket.getaddrinfo(socket.gethostname(), None)
    print("检索到本机IP，提供访问地址")
    pattern = r"(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})"
    use_address = None
    for addr in address:
        match = re.search(pattern, addr[4][0])
        if match:
            print(f"{match.group(1)}")
            use_address = match.group(1)
            break
    app.run(host="0.0.0.0", port=5000, debug=True)
