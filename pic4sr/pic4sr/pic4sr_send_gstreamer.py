#!/usr/bin/env python3

# General purpose
import os
import time
import numpy as np
import cv2
from multiprocessing import Process

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

def resize_image(self, img, image_width, image_height):
    #scale_percent = 25 # percent of original size
    #width = int(img.shape[1] * scale_percent / 100)
    #height = int(img.shape[0] * scale_percent / 100)
    dim = (image_width, image_height)
    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    #print('Resized Dimensions : ',resized.shape)
    return resized

def test_cap_time(self, cap, cap_times):
    start = time.perf_counter()
    ret, frame = cap.read()
    cap_time = time.perf_counter() - start
    cap_times.append(cap_time)
    print('%.1fms' % (cap_time * 1000))
    print(f'Average Speed: {1/np.mean(np.array(cap_times))} fps')
    return ret, frame

def run(self):
    fps = 15
    frame_width = 640
    frame_height = 480
    cap_times = []
    cap = cv2.VideoCapture(0)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
    cap.set(cv2.CAP_PROP_FPS, fps)

    gst_str_rtp =  "appsrc ! videoconvert ! videoscale ! video/x-raw,format=I420,width=640,height=480,framerate=15/1 !  videoconvert ! x264enc tune=zerolatency bitrate=500 speed-preset=superfast ! rtph264pay ! udpsink host=130.192.47.134 port=5000"

    if cap.isOpened() is not True:
        print("Cannot open camera. Exiting.")
        quit()
    out = cv2.VideoWriter(gst_str_rtp, 0, fps, (frame_width, frame_height), True)
    while True:
        ret, frame = cap.read()
        #ret, frame = test_cap_time(cap, cap_times)
        if ret is True:
            # Crop Image
            image_cropped = crop_image(frame)

            # Resize Image
            image_resized = resize_image(image_cropped, 50,50)
            out.write(frame)
        else:
            print("Camera error.")
            time.sleep(10)
    cap.release()

def run2():
    fps = 15
    frame_width = 640
    frame_height = 480
    cap_send = cv2.VideoCapture('videotestsrc ! video/x-raw,framerate=20/1 ! videoscale ! videoconvert ! appsink', cv2.CAP_GSTREAMER)
    out_send = cv2.VideoWriter('appsrc ! videoconvert ! x264enc tune=zerolatency bitrate=500 speed-preset=superfast ! rtph264pay ! udpsink host=127.0.0.1 port=5000',cv2.CAP_GSTREAMER,0, fps, (frame_width,frame_height), True)

    if not cap_send.isOpened() or not out_send.isOpened():
        print('VideoCapture or VideoWriter not opened')
        exit(0)

    while True:
        ret,frame = cap_send.read()

        if ret is True:
            # Crop Image
            image_cropped = crop_image(frame)

            # Resize Image
            image_resized = resize_image(image_cropped)

            out_send.write(frame)

            # cv2.imshow('send', frame)
            # if cv2.waitKey(1)&0xFF == ord('q'):
            #   break
        else:
            print('ERROR: empty frame')
            time.sleep(10)

    cap_send.release()
    out_send.release()


if __name__ == '__main__':
    #r = Process(target=run)
    #r.start()
    #r.join()
    run2()
