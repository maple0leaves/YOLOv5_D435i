import pyrealsense2 as rs
import numpy as np
import cv2


def nothing(x):
    pass


def creatTrackbar():
    # 蓝灯
    # cv.createTrackbar("hmin", "color_adjust", 0, 255, nothing)
    # cv.createTrackbar("hmax", "color_adjust", 250, 255, nothing)
    # cv.createTrackbar("smin", "color_adjust", 0, 255, nothing)
    # cv.createTrackbar("smax", "color_adjust", 143, 255, nothing)
    # cv.createTrackbar("vmin", "color_adjust", 255, 255, nothing)
    # cv.createTrackbar("vmax", "color_adjust", 255, 255, nothing)
    # 红灯
    cv2.createTrackbar("hmin", "color_adjust", 0, 255, nothing)
    cv2.createTrackbar("hmax", "color_adjust", 30, 255, nothing)
    cv2.createTrackbar("smin", "color_adjust", 5, 255, nothing)
    cv2.createTrackbar("smax", "color_adjust", 100, 255, nothing)
    cv2.createTrackbar("vmin", "color_adjust", 255, 255, nothing)
    cv2.createTrackbar("vmax", "color_adjust", 255, 255, nothing)

    # 形态学操作阈值调整
    cv2.createTrackbar("open", "mor_adjust", 1, 30, nothing)
    cv2.createTrackbar("close", "mor_adjust", 15, 30, nothing)
    cv2.createTrackbar("erode", "mor_adjust", 1, 30, nothing)
    cv2.createTrackbar("dilate", "mor_adjust", 3, 30, nothing)

    # 摄像头调整
    # cv.createTrackbar("gamma", "cap_adjust", 100, 200, nothing)

    cv2.createTrackbar("z", "z_adjust", 100, 360, nothing)


def hsv_change(frame):
    hmin = cv2.getTrackbarPos('hmin', 'color_adjust')
    hmax = cv2.getTrackbarPos('hmax', 'color_adjust')
    smin = cv2.getTrackbarPos('smin', 'color_adjust')
    smax = cv2.getTrackbarPos('smax', 'color_adjust')
    vmin = cv2.getTrackbarPos('vmin', 'color_adjust')
    vmax = cv2.getTrackbarPos('vmax', 'color_adjust')

    # gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # cv.imshow("gray", gray)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_hsv = np.array([hmin, smin, vmin])
    upper_hsv = np.array([hmax, smax, vmax])
    mask = cv2.inRange(hsv, lowerb=lower_hsv, upperb=upper_hsv)
    return mask


# Declare pointcloud object, for calculating pointclouds and texture mappings  声明云对象
pc = rs.pointcloud()
# We want the points object to be persistent so we can display the last cloud when a frame drops
points = rs.points()

pipeline = rs.pipeline()  # 创建一个管道
config = rs.config()  # Create a config并配置要流​​式传输的管道。
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 15)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 15)
# 使用选定的流参数显式启用设备流

# Start streaming 开启流
pipe_profile = pipeline.start(config)

# Create an align object 创建对其流对象
# rs.align allows us to perform alignment of depth frames to others frames
# The "align_to" is the stream type to which we plan to align depth frames.
# (对其流)
align_to = rs.stream.color
align = rs.align(align_to)  # 设置为其他类型的流,意思是我们允许深度流与其他流对齐
print(type(align))
cap = cv2.VideoCapture(0)


def led_practice():
    # creatTrackbar()
    while True:

        frames = pipeline.wait_for_frames()  # 等待开启通道
        # ret, frame = cap.read()  # ret 读取到图片为True 未读到图片为Falst
        # frame = cv2.flip(frame, 1)
        aligned_frames = align.process(frames)  # 将深度框和颜色框对齐
        depth_frame = aligned_frames.get_depth_frame()  # ?获得对齐后的帧数深度数据(图)
        color_frame = aligned_frames.get_color_frame()  # ?获得对齐后的帧数颜色数据(图)
        img_color = np.asanyarray(color_frame.get_data())  # 把图像像素转化为数组
        img_depth = np.asanyarray(depth_frame.get_data())  # 把图像像素转化为数组
        # img_color2 = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
        # Intrinsics & Extrinsics
        depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics
        color_intrin = color_frame.profile.as_video_stream_profile().intrinsics
        depth_to_color_extrin = depth_frame.profile.get_extrinsics_to(color_frame.profile)

        # 获取深度传感器的深度标尺
        depth_sensor = pipe_profile.get_device().first_depth_sensor()
        depth_scale = depth_sensor.get_depth_scale()

        # 由深度到颜色
        depth_pixel = [240, 320]  # Random pixel
        depth_point = rs.rs2_deproject_pixel_to_point(depth_intrin, depth_pixel, depth_scale)

        color_point = rs.rs2_transform_point_to_point(depth_to_color_extrin, depth_point)
        color_pixel = rs.rs2_project_point_to_pixel(color_intrin, color_point)
        print('depth: ', color_point)
        print('depth: ', color_pixel)

        pc.map_to(color_frame)
        points = pc.calculate(depth_frame)
        vtx = np.asanyarray(points.get_vertices())  # points.get_vertices() 检索点云的顶点
        tex = np.asanyarray(points.get_texture_coordinates())
        i = 640 * 200 + 200
        print('depth: ', [np.float(vtx[i][0]), np.float(vtx[i][1]), np.float(vtx[i][2])])
        cv2.circle(img_color, (300, 250), 8, [255, 0, 255], thickness=-1)
        # cv2.circle(img_color, (300, 250), 8, [255, 0, 255], thickness=-1)

        cv2.putText(img_color, "Distance/cm:" + str(img_depth[300, 250]), (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                    [255, 0, 255])
        cv2.putText(img_color, "X:" + str(np.float(vtx[i][0])), (80, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, [255, 0, 255])
        cv2.putText(img_color, "Y:" + str(np.float(vtx[i][1])), (80, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, [255, 0, 255])
        cv2.putText(img_color, "Z:" + str(np.float(vtx[i][2])), (80, 160), cv2.FONT_HERSHEY_SIMPLEX, 1, [255, 0, 255])
        # cv2.putText api解释:https://blog.csdn.net/weixin_42039090/article/details/80679935
        cv2.imshow('depth_frame', img_color)
        cv2.imshow("dasdsadsa", img_depth)
        # gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

        # cv2.imshow("frame", frames)
        # mask = hsv_change(img_color)
        # cv2.imshow("frame", mask)
        # cv2.imshow('depth_frame', gray)

        key = cv2.waitKey(1)
        if key == 27:
            cv2.destroyAllWindows()
            break


led_practice()
cv2.waitKey(0)
cv2.destroyAllWindows()
pipeline.stop()
