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

class Pic4sr_cam(Node):
	def __init__(self):
		super().__init__('pic4sr_cam')
		rclpy.logging.set_logger_level('pic4sr_cam', 10)
		self.declare_parameters(namespace='',
		parameters=[
			('sensor', 'rgb'),
			('image_width', 80),
			('image_height', 60),
			('fps', 15)
			])

		self.sensor = self.get_parameter('sensor').get_parameter_value().string_value
		self.image_width = self.get_parameter('image_width').get_parameter_value().integer_value
		self.image_height = self.get_parameter('image_height').get_parameter_value().integer_value
		self.fps = self.get_parameter('fps').get_parameter_value().integer_value

		qos = QoSProfile(depth=10)

		# Create Publishers
		self.color_topic = '/camera/rgb/image_raw'
		self.color_publisher_ = self.create_publisher(Image, self.color_topic, qos)
		self.get_logger().info('start publishing on ' + self.color_topic)
		self.bridge = CvBridge()
		self.cap_times = []
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

	def test_cap_time(self, cap):
		start = time.perf_counter()
		ret, frame = cap.read()
		cap_time = time.perf_counter() - start
		self.cap_times.append(cap_time)
		print('%.1fms' % (cap_time * 1000))
		print(f'Average Speed: {1/np.mean(np.array(self.cap_times))} fps')
		return ret, frame

	def run(self):
		# Opens the inbuilt camera of laptop to capture video.
		cap = cv2.VideoCapture(0)
		while(cap.isOpened()):
			ret, frame = self.test_cap_time(cap)
			# This condition prevents from infinite looping
			# incase video ends.
			if ret == False:
				break
			# Crop Images
			image_cropped = self.crop_image(frame)

			# Resize Images
			image_resized = self.resize_image(image_cropped)
			#self.show_image(image_resized, 'original image')

			# Publish images on ros topics
			self.color_publisher_.publish(self.bridge.cv2_to_imgmsg(image_resized, "bgr8"))
			#self.get_logger().info('Publishing an RGB image')

		cap.release()
		cv2.destroyAllWindows()

def main(args=None):
	rclpy.init()
	pic4sr_cam = Pic4sr_cam()
	pic4sr_cam.get_logger().info('Spinning')
	rclpy.spin(pic4sr_cam)

	pic4sr_cam.destroy_node()
	rclpy.shutdown()

if __name__ == '__main__':
	main()
