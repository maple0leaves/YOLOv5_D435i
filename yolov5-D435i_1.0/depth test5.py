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