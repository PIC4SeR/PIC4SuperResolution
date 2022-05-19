#!/usr/bin/env python3

# General purpose
import time
import numpy as np
import cv2
from multiprocessing import Process


def show_image(image, text):
    colormap = np.asarray(image, dtype = np.uint8)
    cv2.namedWindow(text, cv2.WINDOW_NORMAL)
    cv2.imshow(text,colormap)
    cv2.waitKey(1)

def crop_image( image):
    y = 0
    h = 480
    x = (640-480)//2
    w = 480
    crop_img = image[y:y+h, x:x+w]
    return crop_img

def resize_image(img, image_width, image_height):
    #scale_percent = 25 # percent of original size
    #width = int(img.shape[1] * scale_percent / 100)
    #height = int(img.shape[0] * scale_percent / 100)
    dim = (image_width, image_height)
    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    #print('Resized Dimensions : ',resized.shape)
    return resized

def test_cap_time(cap, cap_times):
    start = time.perf_counter()
    ret, frame = cap.read()
    cap_time = time.perf_counter() - start
    cap_times.append(cap_time)
    print('%.1fms' % (cap_time * 1000))
    print(f'Average Speed: {1/np.mean(np.array(cap_times))} fps')
    return ret, frame

def run():
    host = "127.0.0.1"
    port = 5000

    fps = 30
    # MUST BE MULTIPLE OF 8
    frame_width = 80
    frame_height = 60
    codec = 'JPEG'

    cap_width = 640
    cap_height = 480
    cap_times = []
    cap = cv2.VideoCapture(0)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cap_height)
    cap.set(cv2.CAP_PROP_FPS, fps)

    if codec == 'H264':    
        gst_str_rtp =  ("appsrc "
        "! queue leaky=2 "
        #  "! decodebin "
         "! videoconvert "
         "! videoscale "
         f"! video/x-raw,format=(string)BGR,width=(int){frame_width},height=(int){frame_height},framerate={fps}/1 "
         "! videoconvert "
         "! x264enc tune=zerolatency bitrate=2048 speed-preset=medium quantizer=35 "
         "! rtph264pay "
         "! queue "
         f"! udpsink host={host} port={port}")

    if codec == 'H265':    
        gst_str_rtp =  ("appsrc "
        "! queue leaky=2 "
        #  "! decodebin "
         "! videoconvert "
         "! videoscale "
         f"! video/x-raw,format=(string)I420,width=(int){frame_width},height=(int){frame_height},framerate={fps}/1 "
         "! videoconvert "
         "! x265enc "
         "! rtph265pay "
         "! queue "
         f"! udpsink host={host} port={port}")

    if codec == 'JPEG':
        gst_str_rtp =  ("appsrc "
        "! queue leaky=2 "
        "! decodebin "
        "! videoconvert "
        "! videoscale "
        f"! video/x-raw,format=(string)I420,width={frame_width},height={frame_height},framerate={fps}/1 "
        "! videoconvert "
        "! jpegenc "
        "! rtpjpegpay "
        "! queue leaky=2 "
        f"! udpsink host={host} port={port}")

    if not cap.isOpened():
        print('VideoCapture not opened')
        exit(0)

    out = cv2.VideoWriter(gst_str_rtp, 0, fps, (frame_width, frame_height), True)
    
    if not out.isOpened():
        print('VideoWriter not opened')
        exit(0)
    
    while True:
        ret, frame = cap.read()
        #ret, frame = test_cap_time(cap, cap_times)
        if ret is True:
            # Crop Image
            image_cropped = crop_image(frame)

            # Resize Image
            image_resized = resize_image(image_cropped, frame_width, frame_height)
            out.write(image_resized)
        else:
            print("Camera error.")
            time.sleep(10)
    cap.release()


if __name__ == '__main__':
    #r = Process(target=run)
    #r.start()
    #r.join()
    run()
