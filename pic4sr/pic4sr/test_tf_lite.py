#!/usr/bin/env python3

import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1" 
#import tensorflow as tf

import random
import sys
import time

import numpy as np
import math

import cv2
from cv_bridge import CvBridge
#from pic4sr.utils.model_class_CORAL import ModelCORAL
from pic4sr.utils.model_class_TFlite import ModelTFlite

class Pic4sr_ground():
    def __init__(self):

        model_path = '/root/ros2_ws/src/PIC4SuperResolution/pic4sr/pic4sr/models/srgan'
        image_path = '/root/ros2_ws/src/PIC4SuperResolution/pic4sr/pic4sr/imgs/img.jpg'
        img = cv2.imread(image_path)

        model_path = model_path+'_cpu_'+str(80)+'_'+str(60)+'.tflite'
        self.sr_model = ModelTFlite(model_path)
        self.latencies = []
        self.test_inference_time(img)

    def test_inference_time(self, image):
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        image = np.asarray(image, dtype = np.float32)
        while True:
            start = time.perf_counter()
            output_img = self.sr_model.predict(image)
            inference_time = time.perf_counter() - start
            self.latencies.append(inference_time)
            print('%.1fms' % (inference_time * 1000))
            print(f'Average Speed: {1/np.mean(np.array(self.latencies))} fps')

def main(args=None):

    pic4sr_ground = Pic4sr_ground()

if __name__ == '__main__':
    main()
