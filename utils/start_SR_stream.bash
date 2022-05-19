#!/bin/bash
while getopts e:h:w:i: flag
do
    case "${flag}" in
        e) encoding=${OPTARG};;
        h) height=${OPTARG};;
        w) width=${OPTARG};;
        i) ipaddr=${OPTARG};;
    esac
done

echo "encoding: ${encoding}"
echo "height: ${height}"
echo "width: ${width}"
echo "sending to: ${ipaddr}"

if test $encoding='JPEG'
then
gst-launch-1.0 v4l2src  device=/dev/video0 ! video/x-raw,width=640,height=480,framerate=15/1 \
! queue ! videoconvert ! videoscale ! video/x-raw,width=$width,height=$height ! videoconvert \
! jpegenc ! rtpjpegpay \
! queue leaky=2 ! udpsink host="${ipaddr}" port=5000
fi

if test $encoding='H264'
then
gst-launch-1.0 v4l2src  device=/dev/video0 ! video/x-raw,width=640,height=480,framerate=15/1 \
! queue ! videoconvert ! videoscale ! video/x-raw,width=$width,height=$height ! videoconvert \
! x264enc speed-preset=superfast tune=zerolatency ! rtph264pay \
! queue leaky=2 ! udpsink host="${ipaddr}" port=5000
fi




