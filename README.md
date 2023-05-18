# YOLOv5_D435i
Combined with YOLOv5 to develop Intel_Realsense_D435i to realize real-time detection of three-dimensional distance between objects

结合YOLOv5对Intel_Realsense_D435i 进行开发，实现实时检测物体之间的三维距离

[yolov5](https://github.com/ultralytics/yolov5):实时目标检测算法 

[Intel Relsense D435i深度摄像头](https://www.intelrealsense.com/zh-hans/depth-camera-d435i/):Intel使用realsense(实感)技术开发出来的的深度摄像头，可以获取目标的三维信息
## 1.Use and Environment:
如果您想直接使用，请使用yolov5_D435i_2.0，yolov5_D435i_1.0是本人学习时的版本。
### Environment:
1.一个可运行yolov5的环境
2.一个Intel realsense D435i相机,pyrealsense2和各种依赖库
```
1. could run yolov5
2. pip install -r requirements.txt
3. pip install pyrealsense2
```
  
### Use:
配置yolov5_D435i_2.0/config/yolov5s.yaml,运行yolov5_D435i_2.0/config/main2.py即可

yolov5_D435i_2.0/config/yolov5s.yaml:
```
weight:  "weights/yolov5s.pt"
# 输入图像的尺寸
input_size: [640,480]
# 类别个数
class_num:  80
# 标签名称
class_name: [ 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
         'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
         'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
         'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
         'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
         'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
         'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
         'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
         'hair drier', 'toothbrush' ]
# 阈值设置
threshold:
  iou: 0.45
  confidence: 0.6
# 计算设备
# - cpu
# - 0 <- 使用GPU
device: '0'
target: ['person']#检测哪些类别之间的距离 which objects you want to detect
```
## 2.Attenion
分辨率好像只能改特定的参数，不然会报错。d435i可以用 1280x720, 640x480, 848x480。
```
config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
```
要使用USB3.0接口，不然会报错
yolov5_D435i_2.0/config/yolov5s.yaml中的target是您想要检测距离的类别
## 3.Result
![](https://github.com/maple0leaves/YOLOv5_D435i/blob/master/yolov5_D435i_2.0/image/distance.png)
![](https://github.com/maple0leaves/YOLOv5_D435i/blob/master/yolov5_D435i_2.0/image/distance.gif)
## 4.Reference
[](https://github.com/ultralytics/yolov5)
[](https://github.com/killnice/yolov5-D435i)
[](https://github.com/Thinkin99/yolov5_d435i_detection)
## 5.More Detail
[]()
