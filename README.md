# 人脸识别Demo
有两种模式，视频实时监测，图片识别，可以识别人脸库中对应人脸，及其人脸的表情和性别。

<img src="doc/result.jpg"></img>
## 源码结构

```bash
├── LICENSE
├── README.md
├── doc // 文档资源
├── facerecognize
│   ├── __init__.py
│   ├── img_test.py
│   ├── recognize_imgs // 需要识别的图片
│   ├── train_imgs // 人脸图片库，文件名人物名
│   ├── trained_models
│   │   ├── detection_models 
│   │   ├── emotion_models //表情模型
│   │   └── gender_models //性别模型
│   ├── video_emotion_test.py
│   ├── video_emotion_testv2.py
│   ├── video_emotion_testv3.py
│   └── video_test.py
├── requirements.txt // 运行依赖
└── server
```

## 安装
pip install -r requirements.txt

## 运行
打开video_emotion_testv2.py
运行main函数即可，可以通过如下，切换图片识别还是视频识别
```python
if __name__ == '__main__':
    # live_figout()
    img_figout()
```