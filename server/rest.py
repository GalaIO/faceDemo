# -*- coding: utf-8 -*-
# 
# @Author: gyj176383
# @Date: 2019/5/12

from flask import Flask
from facerecognize import video_emotion_testv3
import json
import threading

app = Flask(__name__)

@app.route('/face/live/0')
def get_person():
    return json.dumps(video_emotion_testv3.live_persons)


class BackgroundTask(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)

    def run(self):
        video_emotion_testv3.live_figout()


def run_backgroud_task():
    BackgroundTask().start()


if __name__ == '__main__':
    # run_backgroud_task()
    # app.debug = True
    app.run(host='0.0.0.0', port=8080)
