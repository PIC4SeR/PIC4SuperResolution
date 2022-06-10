#!/usr/bin/env python3

# General purpose
import time
import numpy as np

# ROS related
import rclpy
from rclpy.node import Node

from std_msgs.msg import String
from std_srvs.srv import Empty
from sensor_msgs.msg import Image

from rclpy.qos import QoSProfile
from rclpy.qos import qos_profile_sensor_data

# others
import cv2
from cv_bridge import CvBridge
import pyrealsense2 as rs

class Pic4sr_realsense(Node):
	def __init__(self):
		super().__init__('pic4sr_realsense')
		rclpy.logging.set_logger_level('pic4sr_realsense', 10)
		self.declare_parameters(namespace='',
		parameters=[
			('sensor', 'rgb'),
			('image_width', 160),
			('image_height', 120),
			('fps', 30)
			])

		self.sensor = self.get_parameter('sensor').get_parameter_value().string_value
		self.image_width = self.get_parameter('image_width').get_parameter_value().integer_value
		self.image_height = self.get_parameter('image_height').get_parameter_value().integer_value
		self.fps = self.get_parameter('fps').get_parameter_value().integer_value

		qos = QoSProfile(depth=10)

		# Configure depth and color streams
		self.pipeline = rs.pipeline()
		self.config = rs.config()

		# Get device product line for setting a supporting resolution
		self.pipeline_wrapper = rs.pipeline_wrapper(self.pipeline)
		self.pipeline_profile = self.config.resolve(self.pipeline_wrapper)
		self.device = self.pipeline_profile.get_device()
		self.device_product_line = str(self.device.get_info(rs.camera_info.product_line))

		self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, self.fps)
		self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, self.fps)

		# Start streaming
		self.pipeline.start(self.config)

		# Create an align object
		# rs.align allows us to perform alignment of depth frames to others frames
		# The "align_to" is the stream type to which we plan to align depth frames.
		self.align_to = rs.stream.color
		self.align = rs.align(self.align_to)
		
		# Create Publishers
		self.color_topic = '/camera/rgb/image_raw'
		self.depth_topic = '/camera/depth/image_raw'
		self.color_publisher_ = self.create_publisher(Image, self.color_topic, qos)
		self.get_logger().info('start publishing on ' + self.color_topic)
		self.depth_publisher_ = self.create_publisher(Image, self.depth_topic, qos)
		self.get_logger().info('start publishing on ' + self.depth_topic)
		self.bridge = CvBridge()
		self.run()
		
	def show_image(self, image, text):
		colormap = np.asarray(image, dtype = np.uint8)
		cv2.namedWindow(text, cv2.WINDOW_NORMAL)
		cv2.imshow(text,colormap)
		cv2.waitKey(1)

	def crop_image(self, image):
		y = 0
		h = 480
		x = (640-480)//2
		w = 480
		crop_img = image[y:y+h, x:x+w]
		return crop_img

	def resize_image(self, img):
		#scale_percent = 25 # percent of original size
		#width = int(img.shape[1] * scale_percent / 100)
		#height = int(img.shape[0] * scale_percent / 100)
		dim = (self.image_width, self.image_height)
		resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
		#print('Resized Dimensions : ',resized.shape)
		return resized

	def run(self):
		try:
			while True:
				# Wait for a coherent pair of frames: depth and color
				frames = self.pipeline.wait_for_frames()
				# Align the depth frame to color frame
				aligned_frames = self.align.process(frames)

				depth_frame = aligned_frames.get_depth_frame()
				color_frame = aligned_frames.get_color_frame()
				if not depth_frame or not color_frame:
				    continue

				# Convert images to numpy arrays
				depth_image = np.asanyarray(depth_frame.get_data())
				color_image = np.asanyarray(color_frame.get_data())

				# Crop Images
				color_cropped = self.crop_image(color_image)
				depth_cropped = self.crop_image(depth_image)

				# Resize Images
				color_resized = self.resize_image(color_cropped)
				depth_resized = self.resize_image(depth_cropped)

				# Publish images on ros topics
				self.color_publisher_.publish(self.bridge.cv2_to_imgmsg(color_resized, "bgr8"))
				#self.get_logger().info('Publishing an RGB image')
				self.depth_publisher_.publish(self.bridge.cv2_to_imgmsg(depth_resized, "16UC1"))
				#self.get_logger().info('Publishing a depth image')

				# Apply colormap on depth image (image must be converted to 8-bit per pixel first)
				#depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

				# depth_colormap_dim = depth_colormap.shape
				# color_colormap_dim = color_image.shape

				# #If depth and color resolutions are different, resize color image to match depth image for display
				# if depth_colormap_dim != color_colormap_dim:
				# 	resized_color_image = cv2.resize(color_image, dsize=(depth_colormap_dim[1], depth_colormap_dim[0]), interpolation=cv2.INTER_AREA)
				# 	images = np.hstack((resized_color_image, depth_colormap))
				# else:
				#  	images = np.hstack((color_image, depth_colormap))

				# #Show images
				# cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
				# cv2.imshow('RealSense', images)
				#cv2.waitKey(1)

			
		finally:

		    # Stop streaming
		    self.pipeline.stop()

def main(args=None):
	rclpy.init()
	pic4sr_realsense = Pic4sr_realsense()
	pic4sr_realsense.get_logger().info('Spinning')
	rclpy.spin(pic4sr_realsense)

	pic4sr_realsense.destroy_node()
	rclpy.shutdown()

if __name__ == '__main__':
	main()
