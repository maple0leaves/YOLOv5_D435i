import pyrealsense2 as rs
import numpy as np
import cv2
import random
import torch
import time
import math

model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
# model = torch.load('yolov5s.pt')
#
# model = torch.hub.load('ultralytics/yolov5', 'yolov5l6')
model.conf = 0.5
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
""""
mean() 函数定义：
numpy.mean(a, axis, dtype, out，keepdims )

mean()函数功能：求取均值
经常操作的参数为axis，以m * n矩阵举例：

axis 不设置值，对 m*n 个数求均值，返回一个实数
axis = 0：压缩行，对各列求均值，返回 1* n 矩阵
axis =1 ：压缩列，对各行求均值，返回 m *1 矩阵
"""
def dectshow(org_img, boxs,depth_data):
    img = org_img.copy()
    for box in boxs:
        if box[6]=='person':#是人才显示
            cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 255), 2)
            dist = get_mid_pos(org_img, box, depth_data, 24)
            cv2.putText(img, box[-1] + str(dist / 1000)[:4] + 'm',
                        (int(box[0]), int(box[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.imshow('dec_img', img)


def juli (i,j,img):
    X = int((boxs[i][2] - boxs[i][0]) / 2 + boxs[i][0])
    Y = int((boxs[i][3] - boxs[i][1]) / 2 + boxs[i][1])
    X1 = int((boxs[j][2] - boxs[j][0]) / 2 + boxs[j][0])
    Y1 = int((boxs[j][3] - boxs[j][1]) / 2 + boxs[j][1])

    dis = aligned_depth_frame.get_distance(X, Y)
    dis1 = aligned_depth_frame.get_distance(X1, Y1)  # （x, y)点的真实深度值
    camera_coordinate = rs.rs2_deproject_pixel_to_point(depth_intrin, [X, Y],
                                                        dis)  # （x, y)点在相机坐标系下的真实值，为一个三维向量。其中camera_coordinate[2]仍为dis，camera_coordinate[0]和camera_coordinate[1]为相机坐标系下的xy真实距离。
    camera_coordinate1 = rs.rs2_deproject_pixel_to_point(depth_intrin, [X1, Y1], dis1)

    # print("0:", camera_coordinate[0], camera_coordinate1[0])
    # print("1:", camera_coordinate[1], camera_coordinate1[1])
    # print("2:", camera_coordinate[2], camera_coordinate1[2])
    # # input()
    a = camera_coordinate[0] - camera_coordinate1[0]
    b = camera_coordinate[1] - camera_coordinate1[1]
    c = camera_coordinate[2] - camera_coordinate1[2]
    # print(a * a, b * b, c * c)
    H = a * a + b * b + c * c
    cv2.line(img,(X,Y),(X1,Y1),(0,255,0),2)
    Q = int((X+X1)/2)
    G = int((Y+Y1)/2)
    H=math.sqrt(H)
    cv2.putText(img,str(H)[:4],(Q,G),cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    print("H", math.sqrt(H))

# def loopDeal(target):






if __name__ == "__main__":
    # Configure depth and color streams

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    # Start streaming
    #pipeline.start(config)
    profile = pipeline.start(config)  # 流程开始
    align_to = rs.stream.color  # 与color流对齐
    align = rs.align(align_to)
    target = 'person'
    try:
        while True:
            # Wait for a coherent pair of frames: depth and color
            frames = pipeline.wait_for_frames()#wait_for_frame/s 自然适用于单个管道/frame_queue 对象。如果您有多个，最好使用 poll_for_frames 以确保在等待另一个队列时不会丢失一个队列中的帧
            aligned_frames = align.process(frames)  # 获取对齐帧
            aligned_depth_frame = aligned_frames.get_depth_frame()  # 获取对齐帧中的depth帧
            color_frame = aligned_frames.get_color_frame()  # 获取对齐帧中的color帧
            depth_intrin = aligned_depth_frame.profile.as_video_stream_profile().intrinsics  # 获取深度参数（像素坐标系转相机坐标系会用到）
            # color_image = np.asanyarray(color_frame.get_data())  # RGB图

            depth_frame = frames.get_depth_frame()
            #color_frame = frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue
            # Convert images to numpy arrays

            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())  # RGB图
            #color_image = np.asanyarray(color_frame.get_data())#这个就是原始的图象

            results = model(color_image)#yolov5权重检测后产生的结果
            #print("results:",results)
            #input()
            boxs= results.pandas().xyxy[0].values
            if len(boxs)==0:
                continue
            for box in boxs:#显示框的中心点
                if box[6]==target:
                    a = (box[2] - box[0]) / 2 + box[0]
                    b = (box[3] - box[1]) / 2 + box[1]
                    cv2.circle(color_image, (int(a), int(b)), 5, (0, 0, 255), -1)

            if len(boxs) !=1 :
                for i in range(len(boxs)-1):
                    for j in range(i+1,len(boxs)):
                        if boxs[i][6] == target and boxs[j][6] == target:
                            # if boxs[i][6] !=boxs[j][6]:
                            juli(i,j,color_image)
                            # X = (boxs[i][2] - boxs[i][0]) / 2 + boxs[i][0]
                            # Y = (boxs[i][3] - boxs[i][1]) / 2 + boxs[i][1]
                            # X1 = (boxs[j][2] - boxs[j][0]) / 2 + boxs[j][0]
                            # Y1 = (boxs[j][3] - boxs[j][1]) / 2 + boxs[j][1]
                            # dis = aligned_depth_frame.get_distance(X, Y)
                            # dis1 = aligned_depth_frame.get_distance(X1, Y1)  # （x, y)点的真实深度值
                            # camera_coordinate = rs.rs2_deproject_pixel_to_point(depth_intrin, [X, Y],
                            # dis)  # （x, y)点在相机坐标系下的真实值，为一个三维向量。其中camera_coordinate[2]仍为dis，camera_coordinate[0]和camera_coordinate[1]为相机坐标系下的xy真实距离。
                            # camera_coordinate1 = rs.rs2_deproject_pixel_to_point(depth_intrin, [X1, Y1], dis1)
                            #
                            # # print("0:", camera_coordinate[0], camera_coordinate1[0])
                            # # print("1:", camera_coordinate[1], camera_coordinate1[1])
                            # # print("2:", camera_coordinate[2], camera_coordinate1[2])
                            # # # input()
                            # a = camera_coordinate[0] - camera_coordinate1[0]
                            # b = camera_coordinate[1] - camera_coordinate1[1]
                            # c = camera_coordinate[2] - camera_coordinate1[2]
                            # #print(a * a, b * b, c * c)
                            # H = a * a + b * b + c * c
                            # print("H", math.sqrt(H))

            print("boxs",boxs,boxs[0][6])
            # input()
            #boxs = np.load('temp.npy',allow_pickle=True)

            dectshow(color_image, boxs, depth_image)

            # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
            #depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
            # Stack both images horizontally
            #images = np.hstack((color_image, depth_colormap))
            # Show images
            # cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
            # cv2.imshow('RealSense', images)
            key = cv2.waitKey(1)
            # Press esc or 'q' to close the image window
            if key & 0xFF == ord('q') or key == 27:
                cv2.destroyAllWindows()
                break
    finally:
        # Stop streaming
        pipeline.stop()
    # loopDeal('person')












