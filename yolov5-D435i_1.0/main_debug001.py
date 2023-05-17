import pyrealsense2 as rs
import numpy as np
import cv2
import random
import torch
import time

model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
#
# model = torch.hub.load('ultralytics/yolov5', 'yolov5l6')
model.conf = 0.5
headers = {'User-Agent': 'User-Agent:Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_3) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/56.0.2924.87 Safari/537.36'}

def get_mid_pos(frame,box,depth_data,randnum):
    distance_list = []
    mid_pos = [(box[0] + box[2])//2, (box[1] + box[3])//2] #确定索引深度的中心像素位置
    min_val = min(abs(box[2] - box[0]), abs(box[3] - box[1])) #确定深度搜索范围
    #print(box,)
    for i in range(randnum):
        bias = random.randint(-min_val//4, min_val//4)
        dist = depth_data[int(mid_pos[1] + bias), int(mid_pos[0] + bias)]
        cv2.circle(frame, (int(mid_pos[0] + bias), int(mid_pos[1] + bias)), 4, (255,0,0), -1)
        #print(int(mid_pos[1] + bias), int(mid_pos[0] + bias))
        if dist:
            distance_list.append(dist)
    distance_list = np.array(distance_list)
    distance_list = np.sort(distance_list)[randnum//2-randnum//4:randnum//2+randnum//4] #冒泡排序+中值滤波
    #print(distance_list, np.mean(distance_list))
    return np.mean(distance_list)
def dectshow(org_img, boxs,depth_data):
    img = org_img.copy()
    for box in boxs:
        cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)
        dist = get_mid_pos(org_img, box, depth_data, 24)
        cv2.putText(img, box[-1] + str(dist / 1000)[:4] + 'm',
                    (int(box[0]), int(box[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.imshow('dec_img', img)

if __name__ == "__main__":
    # Configure depth and color streams
    # set up stream
    pipeline = rs.pipeline()
    config = rs.config()

    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 15)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 15)

    # Start streaming
    profile = pipeline.start(config)

    # Getting the depth sensor's depth scale (see rs-align example for explanation)
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    print("Depth Scale is: ", depth_scale)

    # Create an align object
    # rs.align allows us to perform alignment of depth frames to others frames
    # The "align_to" is the stream type to which we plan to align depth frames.
    align_to = rs.stream.depth
    align = rs.align(align_to)

    try:
        while True:

            start = time.time()  # We want to see how long each iteration takes

            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)

            color_frame = frames.get_color_frame()
            color_image = np.asanyarray(color_frame.get_data())

            depth_frame = aligned_frames.get_depth_frame()
            depth_image = np.asanyarray(depth_frame.get_data())

            if not depth_frame or not color_frame:
                continue

            # Convert images to numpy arrays

            depth_image = np.asanyarray(depth_frame.get_data())

            color_image = np.asanyarray(color_frame.get_data())

            results = model(color_image)
            boxs= results.pandas().xyxy[0].values
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
