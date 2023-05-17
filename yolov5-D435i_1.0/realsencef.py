import numpy as np
import cv2
import matplotlib.pyplot as plt
import random

import pyrealsense2 as rs
pipeline = rs.pipeline()
bagfile = 'realsense/record/20200901.bag'
config = rs.config()
#rs.config.enable_device_from_file(config, bagfile, repeat_playback=False)

config.enable_stream(rs.stream.color, 1280, 720, rs.format.rgb8, 30)
config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)

pf = pipeline.start(config)
# 将深度图对齐到RGB
align_to = rs.stream.color
align = rs.align(align_to)
print(f"align:{align}")

# 获取Realsence一帧的数据
frame = pipeline.wait_for_frames()

# frame的基本元素
print(f"data: {frame.data}")
print(f"frame_number: {frame.frame_number}")
print(f"frame_timestamp_domain: {frame.frame_timestamp_domain}")
print(f"profile: {frame.profile}")
print(f"timestamp: {frame.timestamp}")
# 获取图像数据
data_sz = frame.get_data_size()
print(data_sz, f"1280*720={1280* 720 * 3}")

color_rs = frame.get_color_frame()
img = np.asanyarray(color_rs.get_data())
plt.imshow(img)
plt.axis('off')
plt.show()

depth_rs = frame.get_depth_frame()
depth = np.asanyarray(depth_rs.get_data())
print(depth.shape, depth.dtype)
print(depth[0, 0])
print(f"max(depth):{np.max(depth)}; min(depth:{np.min(depth)})")

# 显示深度图,最大作用距离为4m
dimg_gray = cv2.convertScaleAbs(depth, alpha=255/4000)
dimg = cv2.applyColorMap(dimg_gray, cv2.COLORMAP_JET)

plt.imshow(dimg)
plt.axis('off')
plt.show()
plt.imshow(dimg_gray, cmap='gray')
plt.axis('off')
plt.show()

pc = rs.pointcloud()
pc.map_to(color_rs)
points_rs = pc.calculate(depth_rs)

points = np.asanyarray(points_rs.get_vertices())
print(points.shape, f"848*480 = {848*480}")
print(points[0], type(points[0]))

def calc_dist(xyz):
    x, y, z = xyz
    return (x**2 + y**2 + z**2)**0.5
dists = list(map(calc_dist, points))
dists = np.array(dists).reshape(dimg_gray.shape)

fig = plt.figure()

ax = fig.add_subplot(111)
ax.set_title('colorMap')
plt.imshow(dists)
ax.set_aspect('equal')

cax = fig.add_axes([0.12, 0.1, 0.78, 0.8])
cax.get_xaxis().set_visible(False)
cax.get_yaxis().set_visible(False)
cax.patch.set_alpha(0)
cax.set_frame_on(False)
plt.colorbar(orientation='vertical')
plt.show()

