paddleocr在gpu上的保姆级安装手册
# 一、环境配置
目前在paddle环境安装的时候遇到很多坑，主要问题还是在cudnn的安装上，解决问题的关键是使用conda来安装相应的cudatoolkit和cudnn

## 1. 踩坑记录（血泪教训）
- torch可以正常使用gpu，但是不意味着cuda和cudnn就正常安装了，现在torch gpu版本自带cuda这些了，所以torch可以使用不代表paddle可以使用
- paddle验证通过`import paddle; paddle.utils.run_check()`，但是跑个图片报错`CUDNN_STATUS_NOT_SUPPORTED, conv2d_fusion不被cudnn支持`，未能解决

## 2. conda安装cudatoolkit和cudnn
- 最好新建一个独立的conda环境`conda create --name paddle python=3.9`，防止对原有的环境产生影响
- 在线安装: `conda install cudnn==8.2.1`
- 离线安装: 可以访问[conda下载列表](https://repo.anaconda.com/pkgs/main/linux-64/), 这里也直接给出下载链接: [cudatoolkit-11.3.1-h2bc3f7f_2.tar.bz2](https://repo.anaconda.com/pkgs/main/linux-64/cudatoolkit-11.3.1-h2bc3f7f_2.tar.bz2)和[cudnn-8.2.1-cuda11.3_0.tar.bz2](https://repo.anaconda.com/pkgs/main/linux-64/cudnn-8.2.1-cuda11.3_0.tar.bz2)

## 3. 安装paddlepaddle-gpu | paddleocr | paddlehub
- paddlepaddle：`pip install paddlepaddle-gpu==2.5.2`
- paddleocr: `pip install "paddleocr>=2.0." -i https://mirror.baidu.com/pypi/simple --upgrade PyMuPDF==1.21.0`
- paddlehub: `pip install paddlehub`

## 4. 调用测试
```python
import os
import time
from paddleocr import PaddleOCR

filepath = r"/h01305/projects/PaddleOCR/inference_results/1.jpg"

ocr_model = PaddleOCR(use_angle_cls=True, lang="ch", use_gpu=True, show_log=False,
                det_db_box_thresh=0.1, use_dilation=True,
                det_model_dir='/h01305/projects/PaddleOCR/inference/ch_PP-OCRv4_det_server_infer',
                cls_model_dir='/h01305/projects/PaddleOCR/inference/ch_ppocr_mobile_v2.0_cls_infer',
                rec_model_dir='/h01305/projects/PaddleOCR/inference/ch_PP-OCRv4_rec_server_infer')

t1 = time.time()
for i in range(10):
    result = ocr_model.ocr(img=filepath, det=True, rec=True, cls=True)[0]
t2 = time.time()
print((t2-t1) / 10)
print(result)
```

# 二、paddleocr的使用
1. git clone下paddleocr代码: `git clone https://github.com/PaddlePaddle/PaddleOCR.git`
2. 修改`deploy/hubserving/ocr_system/params.py`中本地模型文件路径
3. 修改`deploy/hubserving/ocr_system/config.json`中参数，主要是use_gpu
4. `hub install deploy/hubserving/ocr_system/`
5. 启动服务端：`CUDA_VISIBLE_DEVICES=0 hub serving start -c deploy/hubserving/ocr_system/config.json`
6. 客户端请求：
```python
import requests
import json
import cv2
import base64

def cv2_to_base64(image):
    data = cv2.imencode('.jpg', image)[1]
    return base64.b64encode(data.tostring()).decode('utf8')

# 发送HTTP请求
data = {'images':[cv2_to_base64(cv2.imread("D:\Project\ocr\\1.jpg"))]}
headers = {"Content-type": "application/json"}
url = "http://10.16.38.1:8868/predict/ocr_system"
r = requests.post(url=url, headers=headers, data=json.dumps(data))

# 打印预测结果
print(r.json())请输入正文
```