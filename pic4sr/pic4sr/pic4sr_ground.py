#!/usr/bin/env python3

import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1" 
#import tensorflow as tf

# gpus = tf.config.experimental.list_physical_devices('GPU')

# if gpus:
# 	try:
# 	# Currently, memory growth needs to be the same across GPUs
# 		for gpu in gpus:
# 			tf.config.experimental.set_memory_growth(gpu, True)
# 			logical_gpus = tf.config.experimental.list_logical_devices('GPU')
# 			print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
# 	except RuntimeError as e:
# 	# Memory growth must be set before GPUs have been initialized
# 		print(e)

import random
import sys
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from rclpy.qos import qos_profile_sensor_data
from std_srvs.srv import Empty
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image
import numpy as np
import math

import cv2
from cv_bridge import CvBridge
from pic4sr.utils.model_class_CORAL import ModelCORAL
from pic4sr.utils.model_class_TFlite import ModelTFlite

class Pic4sr_ground(Node):
	def __init__(self):
		super().__init__('pic4sr_ground')
		# To see debug logs
		#rclpy.logging.set_logger_level('pic4rl_environment', 10)
		self.declare_parameters(namespace='',
		parameters=[
			('model_path', '/home/mauromartini/SR_ws/src/pic4sr/pic4sr/models/srgan'),
			('sensor', 'rgb'),
			('image_width', 50),
			('image_height', 50),
			('device', 'cpu')
			])

		self.model_path = self.get_parameter('model_path').get_parameter_value().string_value
		self.sensor = self.get_parameter('sensor').get_parameter_value().string_value
		self.image_width = self.get_parameter('image_width').get_parameter_value().integer_value
		self.image_height = self.get_parameter('image_height').get_parameter_value().integer_value
		self.device = self.get_parameter('device').get_parameter_value().string_value
		"""************************************************************
		** Initialise ROS publishers and subscribers
		************************************************************"""
		qos = QoSProfile(depth=10)

		self.depth_sub = self.create_subscription(
			Image,
					'/camera/depth/image_raw',
					self.depth_callback,
					qos_profile=qos_profile_sensor_data)

		self.rgb_sub = self.create_subscription(
			Image,
			'/camera/rgb/image_raw',
			self.rgb_callback,
			qos_profile=qos_profile_sensor_data)

		self.cutoff = 6.0
		self.depth_image_raw = np.zeros((self.image_height,self.image_width), np.uint8)
		self.show_img = True
		self.bridge = CvBridge()	

		"""************************************************************
		** Instantiate SUPER RESOLUTION model
		************************************************************"""
		if self.device == 'coral':
			self.model_path = self.model_path+'_converted_int8_edgetpu.tflite'
			self.sr_model = ModelCORAL(self.model_path)
		elif self.device == 'cpu':
			self.model_path = self.model_path+'.tflite'
			self.sr_model = ModelTFlite(self.model_path)
		self.latencies = []	
 
	def depth_callback(self, msg):
		depth_image_raw = np.zeros((self.image_height,self.image_width), np.uint8)
		depth_image_raw = self.bridge.imgmsg_to_cv2(msg, '16UC1')
		depth_image_raw = np.array(depth_image_raw, dtype= np.float32)
		# process depth frame
		depth_image = self.process_depth_image(depth_image_raw)
		# make inference
		sr_depth_image = self.test_inference_time(depth_image) # test inference time
		#sr_rgb_image = self.sr_model.predict(rgb_image_raw) # simple prediction

		# stream image
		self.show_image2(sr_depth_image[0], 'SR DEPTH Image')
		#np.save('/home/maurom/depth_images/depth_image.npy', depth_image_raw)
		#cv2.imwrite('/home/mauromartini/mauro_ws/depth_images/depth_img_raw.png', self.depth_image_raw)
	
	def process_depth_image(self,frame):
		# IF SIMULATION
		#max_depth = self.cutoff # [m]
		# IF REAL CAMERA
		max_depth = self.cutoff*1000 # [mm]

		depth_frame = np.nan_to_num(frame, nan=0.0, posinf=max_depth, neginf=0.0)
		depth_frame = np.minimum(depth_frame, max_depth) # [m] in simulation, [mm] with real camera

		if self.show_img:
			depth_frame = depth_frame/np.amax(depth_frame)
			depth_frame = depth_frame*255.0
			self.show_image(depth_frame,'Depth Image')

		# if 3 channels are needed by the backbone
		depth_frame = np.expand_dims(depth_frame, axis = -1)
		depth_frame = np.tile(depth_frame, (1, 1, 3))
		return depth_frame

	def rgb_callback(self, msg):
		rgb_image_raw = np.zeros((self.image_height,self.image_width,3), np.uint8)
		rgb_image_raw = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
		rgb_image_raw = np.array(rgb_image_raw, dtype=np.float32)
		
		# compute SR inference
		#rgb_image = cv2.cvtColor(rgb_image_raw, cv2.COLOR_BGR2RGB)
		#cv2.imwrite('/home/mauromartini/depth_images/rgb_image.png', rgb_image_raw)
		
		sr_rgb_image = self.test_inference_time(rgb_image_raw) # test inference time
		#sr_rgb_image = self.sr_model.predict(rgb_image_raw) # simple prediction

		# stream image
		if self.show_img:
			self.show_image(rgb_image_raw, 'Raw RGB Image')
			self.show_image2(sr_rgb_image[0], 'SR RGB Image')

	def process_rgb_image(self, img):
		#print('image shape: ', img.shape)
		return image

	def show_image(self, image, text):
		colormap = np.asarray(image, dtype = np.uint8)
		cv2.namedWindow(text, cv2.WINDOW_NORMAL)
		cv2.imshow(text,colormap)
		cv2.waitKey(1)

	def show_image2(self, image, text):
		colormap = np.asarray(image, dtype = np.uint8)
		cv2.namedWindow(text, cv2.WINDOW_NORMAL)
		cv2.imshow(text,colormap)
		cv2.waitKey(1)

	def test_inference_time(self, image):
		start = time.perf_counter()
		output_img = self.sr_model.predict(image)
		inference_time = time.perf_counter() - start
		self.latencies.append(inference_time)
		print('%.1fms' % (inference_time * 1000))
		print(f'Average Speed: {1/np.mean(np.array(self.latencies))} fps')
		return output_img

def main(args=None):
	rclpy.init()
	pic4sr_ground = Pic4sr_ground()
	pic4sr_ground.get_logger().info('Node spinning ...')
	rclpy.spin(pic4sr_ground)

	pic4sr_ground.destroy_node()
	rclpy.shutdown()

if __name__ == '__main__':
	main()
