#!/bin/bash
# 查找正在运行的Python进程的PID
PID=$(ps -ef | grep python | grep -v grep | awk '{print $2}')
# 结束进程
kill $PID

source ./venv/bin/activate

python BridgeGradio.py