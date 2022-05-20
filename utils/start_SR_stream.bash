#!/bin/bash

# Example use
# bash start_SR_stream.bash JPEG 80 60 127.0.0.1

encoding=$1
width=$2
height=$3
ipaddr=$4


echo "encoding: ${encoding}"
echo "frame: ${width}X${height}"
echo "sending to: ${ipaddr}"

if [ $encoding = 'JPEG' ]; then
echo "Streaming JPEG"
gst-launch-1.0 v4l2src  device=/dev/video0 ! video/x-raw,width=640,height=480,framerate=15/1 \
! queue ! videoconvert ! videoscale ! video/x-raw,width=$width,height=$height ! videoconvert \
! jpegenc quality=100 idct-method=float ! rtpjpegpay \
! queue leaky=2 ! udpsink host="${ipaddr}" port=5000
fi

if [ $encoding = 'H264' ]; then
echo "Streaming H264"
gst-launch-1.0 v4l2src  device=/dev/video0 ! video/x-raw,width=640,height=480,framerate=15/1 \
! queue ! videoconvert ! videoscale ! video/x-raw,width=$width,height=$height ! videoconvert \
! x264enc speed-preset=superfast tune=zerolatency ! rtph264pay \
! queue leaky=2 ! udpsink host="${ipaddr}" port=5000
fi

if [ $encoding = 'DEV' ]; then
echo "Streaming DEV"
gst-launch-1.0 v4l2src  device=/dev/video0 ! video/x-raw,width=640,height=480,framerate=15/1 \
! queue ! videoconvert ! videoscale ! video/x-raw,width=$width,height=$height ! videoconvert \
! avenc_mjpeg ! rtpjpegpay \
! udpsink host="${ipaddr}" port=5000
fi

