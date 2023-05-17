import pyrealsense2 as rs
import numpy as np
import cv2
import random
import torch
import time
import math
import yaml
#D435i要使用usb3.0接口，如果是usb2.0接口的话会报错  frame didn't arrive within 5000
from Yolov5Model import YoloV5
#感谢开源，可以学习到优秀的人写的代码，让自己的代码也可以写的漂亮一点

#如果想要检测不同类别物体之间的距离怎么办？首先绘制检测框是没有问题，把id列表传进detect中就行了
#距离的话也是用id列表判断
# model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
# model = torch.load('yolov5s.pt')
#
# model = torch.hub.load('ultralytics/yolov5', 'yolov5l6')
# model.conf = 0.5
# headers = {'User-Agent': 'User-Agent:Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_3) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/56.0.2924.87 Safari/537.36'}

#开启流和配置相机参数
# pipeline = rs.pipeline()
# config = rs.config()
# config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
# config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
# # Start streaming
# #pipeline.start(config)
# profile = pipeline.start(config)  # 流程开始
# align_to = rs.stream.color  # 与color流对齐
# align = rs.align(align_to)
#author 221 这个是最终可用版，可以检测1个及以上类别之间的距离
#需要使用的时候只需要修改yolov5syaml文件即可
#明天用一上午注释这个文件，用STAR法则介绍一下你这个项目写成文档，然后整理上传到github
'''
def get_mid_pos(frame,box,depth_data,randnum):
    distance_list = []
    mid_pos = [(box[0] + box[2])//2, (box[1] + box[3])//2] #确定索引深度的中心像素位置
    #print(mid_pos)mid_pos里面的数据是浮点数，不是整数，cv2.circle中的坐标要整数

    min_val = min(abs(box[2] - box[0]), abs(box[3] - box[1])) #确定深度搜索范围
    #print(box,)
    #input()
    """
    random.randint(参数1，参数2)

    参数1、参数2必须是整数
    函数返回参数1和参数2之间的任意整数， 闭区间
    """
    for i in range(randnum):
        bias = random.randint(-min_val//4, min_val//4)
        dist = depth_data[int(mid_pos[1] + bias), int(mid_pos[0] + bias)]
        cv2.circle(frame, (int(mid_pos[0] + bias), int(mid_pos[1] + bias)), 4, (255,0,0), -1)
        #print(int(mid_pos[1] + bias), int(mid_pos[0] + bias))
        if dist:
            distance_list.append(dist)
    distance_list = np.array(distance_list)
    distance_list = np.sort(distance_list)[randnum//2-randnum//4:randnum//2+randnum//4] #冒泡排序+中值滤波
    #print("distance_list, np.mean(distance_list):",distance_list, np.mean(distance_list))
    return np.mean(distance_list)

'''

""""
mean() 函数定义：
numpy.mean(a, axis, dtype, out，keepdims )

mean()函数功能：求取均值
经常操作的参数为axis，以m * n矩阵举例：

axis 不设置值，对 m*n 个数求均值，返回一个实数
axis = 0：压缩行，对各列求均值，返回 1* n 矩阵
axis =1 ：压缩列，对各行求均值，返回 m *1 矩阵
"""
'''
def dectshow(org_img, boxs,depth_data,index):
    img = org_img.copy()
    for box in boxs:
        if box[5]==index:#是index才显示
            cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 255), 2)
            dist = get_mid_pos(org_img, box, depth_data, 24)
            cv2.putText(img, box[-1] + str(dist / 1000)[:4] + 'm',
                        (int(box[0]), int(box[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.imshow('dec_img', img)

'''
#使用自己训练的权重，可以自定义类别，检测类别之间的距离

def juli (i,j,img,depth_intrin):
    #获取二维中心点坐标
    X = int((boxs[i][2] + boxs[i][0]) / 2 )
    Y = int((boxs[i][3] + boxs[i][1]) / 2)
    X1 = int((boxs[j][2] + boxs[j][0]) / 2 )
    Y1 = int((boxs[j][3] + boxs[j][1]) / 2 )
    #使用中心坐标获取中心坐标距离相机的三维距离
    dis = aligned_depth_frame.get_distance(X,Y)
    dis1 = aligned_depth_frame.get_distance(X1,Y1)  # （x, y)点的真实深度值
    #获取在相机坐标系的三维坐标
    camera_coordinate = rs.rs2_deproject_pixel_to_point(depth_intrin, [X,Y],
                                                        dis)  # （x, y)点在相机坐标系下的真实值，为一个三维向量。其中camera_coordinate[2]仍为dis，camera_coordinate[0]和camera_coordinate[1]为相机坐标系下的xy真实距离。
    camera_coordinate1 = rs.rs2_deproject_pixel_to_point(depth_intrin, [X1,Y1], dis1)

    # print("0:", camera_coordinate[0], camera_coordinate1[0])
    # print("1:", camera_coordinate[1], camera_coordinate1[1])
    # print("2:", camera_coordinate[2], camera_coordinate1[2])
    # # input()
    a = camera_coordinate[0] - camera_coordinate1[0]
    b = camera_coordinate[1] - camera_coordinate1[1]
    c = camera_coordinate[2] - camera_coordinate1[2]
    # print(a * a, b * b, c * c)
    H = a * a + b * b + c * c
    #画两个二维中心坐标之间的线
    cv2.line(img,(X,Y),(X1,Y1),(0,255,0),2)
    Q = int((X+X1)/2)
    G = int((Y+Y1)/2)
    H=math.sqrt(H)
    #H 两个中心之间的三维距离
    cv2.putText(img,str(H)[:4],(Q,G),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    # print("H", math.sqrt(H))


if __name__ == "__main__":
    # Configure depth and color streams
    model = YoloV5(yolov5_yaml_path='config/yolov5s.yaml')
    pipeline = rs.pipeline()
    #config 配置流式传输的管道
    config = rs.config()
    #只能改特定尺寸的图片，不然会报错
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    # Start streaming
    #pipeline.start(config)
    profile = pipeline.start(config)  # 流程开始
    # 直接读取出来的彩色图片和深度图片是没有对齐的，读取出来的两张图片像素之间没有一一对应
    #对齐之后，可以直接使用深度信息，即点云
    align_to = rs.stream.color  # 与color流对齐
    align = rs.align(align_to)

    #获取对应类别index
    with open('config/yolov5s.yaml', 'r', encoding='utf-8') as f:
        yolov5yaml = yaml.load(f.read(), Loader=yaml.SafeLoader)
    list = yolov5yaml['class_name']
    target = yolov5yaml['target']
    index = []
    for t in target:
        index.append(list.index(t))
    index=np.array(index)

    #开始读取帧
    try:
        while True:
            # Wait for a coherent pair of frames: depth and color
            # frames = pipeline. try_wait_for_frames ()
            #等待所有配置的流生成框架
            frames = pipeline.wait_for_frames()#wait_for_frame/s 自然适用于单个管道/frame_queue 对象。如果您有多个，最好使用 poll_for_frames 以确保在等待另一个队列时不会丢失一个队列中的帧
            aligned_frames = align.process(frames)  # 获取对齐帧
            #获取对齐帧，在对齐帧里面有depth帧，color帧，深度参数
            aligned_depth_frame = aligned_frames.get_depth_frame()  # 获取对齐帧中的depth帧
            color_frame = aligned_frames.get_color_frame()  # 获取对齐帧中的color帧
            depth_intrin = aligned_depth_frame.profile.as_video_stream_profile().intrinsics  # 获取深度参数（像素坐标系转相机坐标系会用到）
            # color_image = np.asanyarray(color_frame.get_data())  # RGB图
            #深度和彩色框架，通过框架获取图片data
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue
            # Convert images to numpy arrays

            # depth_image = np.asanyarray(depth_frame.get_data())
            # cv2.imshow('depth',depth_image)
            color_image = np.asanyarray(color_frame.get_data())  # RGB图
            #color_image = np.asanyarray(color_frame.get_data())#这个就是原始的图象

            img,_,_,_,results = model.detect(color_image,id=index)#yolov5权重检测后产生的结果
            #print("results:",results)
            #input()
            #转成np更快
            boxs= results.cpu().numpy()
            #过滤出需要的类别
            list=[]
            for box in boxs:
                if box[5]in index:
                    list.append(box)
            boxs = np.array(list)#转换成numppy，更快一点

            if len(boxs)!=0:
                # 显示框的中心点
                for box in boxs:
                        a = (box[2] + box[0]) / 2
                        b = (box[3] + box[1]) / 2
                        cv2.circle(img, (int(a), int(b)), 5, (0, 0, 255), -1)
                #计算距离并显示
                if len(boxs) !=1 :
                    for i in range(len(boxs)-1):
                        for j in range(i+1,len(boxs)):
                                juli(i,j,img,depth_intrin)
            #没有目标则显示原图，有目标显示处理之后图片
            last = color_image if len(boxs)==0 else img
            cv2.imshow('detect',last)
            key = cv2.waitKey(1)
            # Press esc or 'q' to close the image window
            if key & 0xFF == ord('q') or key == 27:
                cv2.destroyAllWindows()
                break
    finally:
        # Stop streaming
        pipeline.stop()
    # loopDeal('person')












