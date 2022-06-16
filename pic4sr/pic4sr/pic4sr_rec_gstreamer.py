#!/usr/bin/env python3

import os

import matplotlib.pyplot as plt

from pic4sr_send_gstreamer import crop_image, resize_image

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

import sys
import time
import numpy as np

import cv2
from utils.model_class_CORAL import ModelCORAL
from utils.model_class_TFlite import ModelTFlite

class Pic4sr_ground():
    def __init__(self):

        self.model_path = '/home/marco/ros2_ws/src/PIC4SuperResolution/pic4sr/pic4sr/models/srgan'
        self.sensor = 'bgr'
        self.image_width = 60
        self.image_height = 80
        self.codec='H264'
        self.device = 'cpu'

        self.cutoff = 6.0
        self.show_img = True	

        """************************************************************
        ** Instantiate SUPER RESOLUTION model
        ************************************************************"""
        if self.device == 'coral':
            self.model_path = self.model_path+'_converted_int8_edgetpu'+str(self.image_width)+str(self.image_height)+'.tflite'
            self.sr_model = ModelCORAL(self.model_path)
        elif self.device == 'cpu':
            self.model_path = self.model_path+str(self.image_width)+str(self.image_height)+'.tflite'
            self.sr_model = ModelTFlite(self.model_path)
        self.latencies = []	
        self.latencies_rec = []

    def crop_image(self, image, h, w):
        y = 2
        x = 0
        crop_img = image[y:y+h, x:x+w]
        return crop_img

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

        # make inference
        sr_depth_image = self.test_inference_time(depth_image) # test inference time
        #sr_rgb_image = self.sr_model.predict(rgb_image_raw) # simple prediction

        # stream image
        self.show_image2(sr_depth_image[0], 'SR DEPTH Image')
        return sr_depth_image

    def process_rgb_image(self, img):
        rgb_image_raw = np.array(img, dtype=np.float32)
        # rgb_image_raw = self.crop_image(rgb_image_raw, self.image_height, self.image_width)

        # rgb_image_raw = np.transpose(rgb_image_raw, (1,0,2))
        # compute SR inference
        #rgb_image = cv2.cvtColor(rgb_image_raw, cv2.COLOR_BGR2RGB)
        #cv2.imwrite('/home/mauromartini/depth_images/rgb_image.png', rgb_image_raw)

        # compute SR inference
        sr_rgb_image = self.test_inference_time(rgb_image_raw) # test inference time
        #sr_rgb_image = self.sr_model.predict(rgb_image_raw) # simple prediction

        # stream image
        if self.show_img:
            # self.show_image(np.transpose(rgb_image_raw, (1,0,2)), 'Raw RGB Image')
            # self.show_image2(np.transpose(sr_rgb_image[0], (1,0,2)), 'SR RGB Image')
            self.show_image(rgb_image_raw, 'Raw RGB Image')
            self.show_image2(sr_rgb_image[0], 'SR RGB Image')
        # print('image shape: ', img.shape)
        return img

    def show_image(self, image, text):
        colormap = np.asarray(image, dtype = np.uint8)
        colormap = cv2.rotate(colormap, cv2.ROTATE_90_COUNTERCLOCKWISE)
        cv2.namedWindow(text, cv2.WINDOW_NORMAL)
        cv2.imshow(text,colormap)
        cv2.waitKey(1)

    def show_image2(self, image, text):
        colormap = np.asarray(image, dtype = np.uint8)
        colormap = cv2.rotate(colormap, cv2.ROTATE_90_COUNTERCLOCKWISE)
        cv2.namedWindow(text, cv2.WINDOW_NORMAL)
        cv2.imshow(text,colormap)
        cv2.waitKey(1)

    def test_inference_time(self, image):
        start = time.perf_counter()
        output_img = self.sr_model.predict(image)
        inference_time = time.perf_counter() - start
        self.latencies.append(inference_time)
        print('%.1fms' % (inference_time * 1000))
        print(f'Average Inf Speed: {1/np.mean(np.array(self.latencies))} fps')
        return output_img

    def rec_frequency(self):
        start = time.perf_counter()
        tdiff = start - self.prev
        self.latencies_rec.append(tdiff)
        self.prev = start
        print(f'Average Rec Speed: {1/np.mean(np.array(self.latencies_rec))} fps')        
        if len(self.latencies_rec) > 200:
            print(f'Average Rec Speed: {1/np.mean(np.array(self.latencies_rec))} fps')        
            print(f'Max Rec Latency: {max(self.latencies_rec)}')
            plt.hist(np.array(self.latencies_rec))
            plt.savefig(os.path.expanduser('~/pic4sr_test/img.png'))
            plt.show()
            exit(0)

    def run(self,):

        if self.codec == 'H264':
            cap_receive = cv2.VideoCapture(
            ("udpsrc port=5000 "
            "! queue leaky=2 " 
            "! application/x-rtp, media=(string)video, clock-rate=(int)90000, encoding-name=(string)H264, payload=(int)96 "
            "! rtph264depay "
            "! h264parse "
            "! avdec_h264 "
            "! decodebin "
            "! videoconvert "
            "! queue leaky=2 "
            "! appsink "), cv2.CAP_GSTREAMER)

        if self.codec == 'H265':
            cap_receive = cv2.VideoCapture(
            ("udpsrc port=5000 "
            "! queue leaky=2 " 
            "! application/x-rtp, media=(string)video, clock-rate=(int)90000, encoding-name=(string)H265, payload=(int)96 "
            "! rtph265depay "
            "! h265parse "
            "! avdec_h265 "
            "! decodebin "
            #   "! videoconvert "
            #   "! video/x-raw,format=(string)BGR "
            "! videoconvert "
            "! appsink "), cv2.CAP_GSTREAMER)

        if self.codec == 'JPEG':
            cap_receive = cv2.VideoCapture(
                ("udpsrc port=5000 "
                # "! queue leaky=2" # lasciare qui
                "! application/x-rtp, encoding-name=(string)JPEG, payload=(int)26 "
                "! rtpjpegdepay "
                "! jpegdec "
                "! decodebin "
                "! videoconvert "
                "! video/x-raw,format=(string)BGR "
                "! videoconvert "
                "! appsink drop=1 "), cv2.CAP_GSTREAMER)

        if not cap_receive.isOpened():
            print('VideoCapture not opened')
            exit(0)
        
        print("Start VideoCapture...")

        self.prev = time.perf_counter()
        while True:
            
            ret,frame = cap_receive.read()  
            # self.show_image(frame, 'rec')
            # self.rec_frequency()


            if ret:
                image = self.process_rgb_image(frame)
            else:
                print('ERROR: empty frame')
                time.sleep(10)

            if cv2.waitKey(1)&0xFF == ord('q'):
                break

        cap_receive.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    pic4sr_ground = Pic4sr_ground()
    pic4sr_ground.run()
