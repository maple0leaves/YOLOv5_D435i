'''
import pyrealsense2 as rs
import numpy as np

pipeline = rs.pipeline()
pc = rs.pointcloud()

config = rs.config()

# This is the minimal recommended resolution for D435
# config.enable_stream(rs.stream.depth,  848, 480, rs.format.z16, 90)
# config.enable_stream(rs.stream.color, 848, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 15)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 15)

pipe_profile = pipeline.start(config)
frames = pipeline.wait_for_frames()
depth_frame = frames.get_depth_frame()
color_frame = frames.get_color_frame()

# Intrinsics & Extrinsics
depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics
print(depth_intrin)
# width: 640, height: 480, ppx: 321.388, ppy: 240.675, fx: 385.526, fy: 385.526, model: Brown Conrady, coeffs: [0, 0, 0, 0, 0]
color_intrin = color_frame.profile.as_video_stream_profile().intrinsics
depth_to_color_extrin = depth_frame.profile.get_extrinsics_to(color_frame.profile)
#
# Depth scale - units of the values inside a depth frame, i.e how to convert the value to units of 1 meter
dist_to_center = depth_frame.get_distance(320, 240)
depth_sensor = pipe_profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
print(depth_scale)
#
# Map depth to color
depth_pixel = [320, 240]  # Random pixel
depth_point = rs.rs2_deproject_pixel_to_point(depth_intrin, depth_pixel, dist_to_center)
print(depth_point)

color_point = rs.rs2_transform_point_to_point(depth_to_color_extrin, depth_point)
# print(color_point)
# [0.014667518436908722, 0.0005329644773155451, 0.0011108629405498505]
color_pixel = rs.rs2_project_point_to_pixel(color_intrin, color_point)
# print(color_pixel)
# [8431.0703125, 534.9332885742188]
pipeline.stop()
'''

import pyrealsense2 as rs
import numpy as np
import cv2
import json
import math

def get_aligned_images():
    frames = pipeline.wait_for_frames()  # 等待获取图像帧
    aligned_frames = align.process(frames)  # 获取对齐帧
    aligned_depth_frame = aligned_frames.get_depth_frame()  # 获取对齐帧中的depth帧
    color_frame = aligned_frames.get_color_frame()  # 获取对齐帧中的color帧

    ############### 相机参数的获取 #######################
    #intr = color_frame.profile.as_video_stream_profile().intrinsics  # 获取相机内参
    depth_intrin = aligned_depth_frame.profile.as_video_stream_profile().intrinsics  # 获取深度参数（像素坐标系转相机坐标系会用到）
    # camera_parameters = {'fx': intr.fx, 'fy': intr.fy,
    #                      'ppx': intr.ppx, 'ppy': intr.ppy,
    #                      'height': intr.height, 'width': intr.width,
    #                      'depth_scale': profile.get_device().first_depth_sensor().get_depth_scale()
    #                      }
    # # 保存内参到本地
    # with open('./intrinsics.json', 'w') as fp:
    #     json.dump(camera_parameters, fp)
    #######################################################

    #depth_image = np.asanyarray(aligned_depth_frame.get_data())  # 深度图（默认16位）
    #depth_image_8bit = cv2.convertScaleAbs(depth_image, alpha=0.03)  # 深度图（8位）
    #depth_image_3d = np.dstack((depth_image_8bit, depth_image_8bit, depth_image_8bit))  # 3通道深度图
    color_image = np.asanyarray(color_frame.get_data())  # RGB图

    # 返回相机内参、深度参数、彩色图、深度图、齐帧中的depth帧
    #return intr, depth_intrin, color_image, depth_image, aligned_depth_frame
    return  depth_intrin, color_image,aligned_depth_frame

if __name__ == "__main__":

    pipeline = rs.pipeline()  # 定义流程pipeline
    config = rs.config()  # 定义配置config
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)  # 配置depth流
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)  # 配置color流
    profile = pipeline.start(config)  # 流程开始
    align_to = rs.stream.color  # 与color流对齐
    align = rs.align(align_to)

    while 1:
        #intr, depth_intrin, rgb, depth, aligned_depth_frame = get_aligned_images()  # 获取对齐的图像与相机内参
        depth_intrin,rgb,aligned_depth_frame = get_aligned_images()  # 获取对齐的图像与相机内参
        # 定义需要得到真实三维信息的像素点（x, y)，本例程以中心点为例
        x = 320
        y = 240
        x1 =200
        y1 =200

        dis = aligned_depth_frame.get_distance(x, y)
        dis1 = aligned_depth_frame.get_distance(x1, y1)  # （x, y)点的真实深度值
        camera_coordinate= rs.rs2_deproject_pixel_to_point(depth_intrin, [x, y],
                                                            dis)  # （x, y)点在相机坐标系下的真实值，为一个三维向量。其中camera_coordinate[2]仍为dis，camera_coordinate[0]和camera_coordinate[1]为相机坐标系下的xy真实距离。
        camera_coordinate1 = rs.rs2_deproject_pixel_to_point(depth_intrin, [x1, y1],dis1)

        print("0:",camera_coordinate[0],camera_coordinate1[0])
        print("1:",camera_coordinate[1],camera_coordinate1[1])
        print("2:",camera_coordinate[2],camera_coordinate1[2])
        #input()
        a=camera_coordinate[0]-camera_coordinate1[0]
        b=camera_coordinate[1]-camera_coordinate1[1]
        c=camera_coordinate[2]-camera_coordinate1[2]
        print(a*a,b*b,c*c)
        H =a*a+b*b+c*c
        print("H",math.sqrt(H))
        #print(camera[2],camera1[2])
        # H = math.pow(camera[2]-camera1[2],2)
        # #print(H)
        # juli =math.sqrt(H)
        # print(juli)
        #print(camera_coordinate[0],camera_coordinate1[0])

        cv2.circle(rgb, (x, y), 5, (0, 0, 255),-1)
        cv2.circle(rgb, (x1, y1), 5, (0, 0, 255), -1)
        cv2.imshow('RGB image', rgb)  # 显示彩色图像
        key = cv2.waitKey(1)
        # Press esc or 'q' to close the image window
        if key & 0xFF == ord('q') or key == 27:
            pipeline.stop()
            break
    cv2.destroyAllWindows()

'''
import cv2
cap=cv2.VideoCapture(1)
while(1):
    a,b=cap.read()
    cv2.circle(b, (447, 63), 10, (0, 0, 255),-1)
    cv2.circle(b, (345, 63), 10, (0, 0, 255),-1)

    cv2.imshow("sdf",b)
    if cv2.waitKey(1)==ord("q"):
        break
cap.release()
cv2.destroyAllWindows()

'''
