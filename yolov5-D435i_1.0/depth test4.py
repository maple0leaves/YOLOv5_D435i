# First import the library
import pyrealsense2 as rs

pipeline = rs.pipeline()
"""
# Create a context object. This object owns the handles to all connected realsense devices
# 创建pipeline对象
# The caller can provide a context created by the application, usually for playback or testing purposes.
"""

pipeline.start()
"""
start(*args, **kwargs)
Overloaded function.

1. start(self: pyrealsense2.pyrealsense2.pipeline, config: rs2::config) -> rs2::pipeline_profile

Start the pipeline streaming according to the configuraion. The pipeline streaming loop captures samples from the 
device, and delivers them to the attached computer vision modules and processing blocks, according to each module 
requirements and threading model. During the loop execution, the application can access the camera streams by calling 
wait_for_frames() or poll_for_frames(). The streaming loop runs until the pipeline is stopped. Starting the pipeline 
is possible only when it is not started. If the pipeline was started, an exception is raised（引发异常）. The pipeline 
selects and activates the device upon start, according to configuration or a default configuration. When the 
rs2::config is provided to the method, the pipeline tries to activate the config resolve() result. If the application 
requests are conflicting with pipeline computer vision modules or no matching device is available on the platform, 
the method fails. Available configurations and devices may change between config resolve() call and pipeline start, 
in case devices are connected or disconnected, or another application acquires ownership of a device. 

2. start(self: pyrealsense2.pyrealsense2.pipeline) -> rs2::pipeline_profile

Start the pipeline streaming with its default configuration. The pipeline streaming loop captures samples from the 
device, and delivers them to the attached computer vision modules and processing blocks, according to each module 
requirements and threading model. During the loop execution, the application can access the camera streams by calling 
wait_for_frames() or poll_for_frames(). The streaming loop runs until the pipeline is stopped. Starting the pipeline 
is possible only when it is not started. If the pipeline was started, an exception is raised. 


3. start(self: pyrealsense2.pyrealsense2.pipeline, callback: Callable[[pyrealsense2.pyrealsense2.frame], 
None]) -> rs2::pipeline_profile 

Start the pipeline streaming with its default configuration.
The pipeline captures samples from the device, and delivers them to the through the provided frame callback.
Starting the pipeline is possible only when it is not started. If the pipeline was started, an exception is raised.
When starting the pipeline with a callback both wait_for_frames() and poll_for_frames() will throw exception.

4. start(self: pyrealsense2.pyrealsense2.pipeline, config: rs2::config, callback: Callable[[
pyrealsense2.pyrealsense2.frame], None]) -> rs2::pipeline_profile 

Start the pipeline streaming according to the configuraion. The pipeline captures samples from the device, 
and delivers them to the through the provided frame callback. Starting the pipeline is possible only when it is not 
started. If the pipeline was started, an exception is raised. When starting the pipeline with a callback both 
wait_for_frames() and poll_for_frames() will throw exception. The pipeline selects and activates the device upon 
start, according to configuration or a default configuration. When the rs2::config is provided to the method, 
the pipeline tries to activate the config resolve() result. If the application requests are conflicting with pipeline 
computer vision modules or no matching device is available on the platform, the method fails. Available 
configurations and devices may change between config resolve() call and pipeline start, in case devices are connected 
or disconnected, or another application acquires ownership of a device. """

try:
    while True:
        # Create a pipeline object. This object configures the streaming camera and owns it's handle
        frames = pipeline.wait_for_frames()
        """wait_for_frames(self: pyrealsense2.pyrealsense2.pipeline, timeout_ms: int=5000) -> 
        pyrealsense2.pyrealsense2.composite_frame 

        Wait until a new set of frames becomes available. The frames set includes time-synchronized frames of each 
        enabled stream in the pipeline. In case of（若在......情况下） different frame rates of the streams, the frames set 
        include a matching frame of the slow stream, which may have been included in previous frames set. The method 
        blocks（阻塞） the calling thread, and fetches（拿来、取来） the latest unread frames set. Device frames, which were 
        produced while the function wasn't called, are dropped（被扔掉）. To avoid frame drops（丢帧、掉帧）, this method should 
        be called as fast as the device frame rate. The application can maintain the frames handles to defer（推迟） 
        processing. However, if the application maintains too long history, the device may lack memory resources to 
        produce new frames, and the following call to this method shall fail to retrieve（检索、取回） new frames, 
        until resources become available. """
        depth = frames.get_depth_frame()
        """
        get_depth_frame(self: pyrealsense2.pyrealsense2.composite_frame) -> rs2::depth_frame

        Retrieve the first depth frame, if no frame is found, return an empty frame instance.
        """
        print(type(frames))
        # <class 'pyrealsense2.pyrealsense2.composite_frame'>
        print(type(depth))
        # <class 'pyrealsense2.pyrealsense2.depth_frame'>
        print(frames)
        # <pyrealsense2.pyrealsense2.composite_frame object at 0x000001E4D0AAB7D8>
        print(depth)
        # <pyrealsense2.pyrealsense2.depth_frame object at 0x000001E4D0C4B228>

        # 如果没有接收到深度帧，跳过执行下一轮循环
        if not depth:
            continue
        print('not depth:', not depth)
        # not depth: False
        # 如果 depth 为空（False），则 not depth 为True，如果 depth 不为空（True），则 not depth 为False

        # Print a simple text-based representation of the image, by breaking it into 10x20 pixel regions and
        # approximating the coverage of pixels within one meter
        coverage = [0] * 64
        print(type(coverage))
        # <class 'list'>
        print(coverage)
        # [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        # 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

        for y in range(480):
            for x in range(640):
                # 获取当前深度图像（x, y）坐标像素的深度数据
                dist = depth.get_distance(x, y)
                """
                get_distance(self: pyrealsense2.pyrealsense2.depth_frame, x: int, y: int) -> float

                Provide the depth in meters at the given pixel
                """
                # 如果当前坐标（x, y）像素的深度在1m范围以内，将其所负责的列表元素变量加1。（如：x在0到9范围内负责列表元素coverage[0]）
                if 0 < dist and dist < 1:
                    # x方向上每10个像素宽度整合为一个新的像素区域（最后整合成 640/10=64 个新像素值），将符合深度要求的点加起来作统计。
                    coverage[x // 10] += 1
            # y方向上每20个像素宽度整合为一个新的像素区域（最后整合成 480/20=24 个新像素值）
            if y % 20 is 19:
                line = ""
                # coverage 列表中元素最大值为200（该区域内【10×20】所有像素点都在所给深度范围内）
                for c in coverage:
                    # c//25的最大值为8
                    # 用所占颜色空间由小到大的文本来近似复现深度图像
                    line += " .:nhBXWW"[c // 25]
                # 重置coverage列表
                coverage = [0] * 64
                print(line)

finally:
    pipeline.stop()

