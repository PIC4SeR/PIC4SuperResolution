#!/bin/bash

sudo apt update

if [ -z "$(ls /usr/bin | grep cmake)" ]; then
  echo "Installing cmake"
  sudo apt install cmake
fi

if [ ! -z "$(python2 -c "import numpy")" ]; then
  echo "Installing numpy"
  sudo apt install python-numpy
fi

# INSTALL GStreamer
if [ -z "$(ls /usr/bin | grep gst)" ]; then
  echo "Installing GStreamer"
  sudo apt-get install libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev libgstreamer-plugins-bad1.0-dev gstreamer1.0-plugins-base gstreamer1.0-plugins-good gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly gstreamer1.0-libav gstreamer1.0-doc gstreamer1.0-tools gstreamer1.0-x gstreamer1.0-alsa gstreamer1.0-gl gstreamer1.0-gtk3 gstreamer1.0-qt5 gstreamer1.0-pulseaudio -y
else
  echo "GStreamer found"
fi
 

if [ ! -d ~/OpenCV ]; then
  mkdir ~/OpenCV
  echo "Source directory at ~/OpenCV"
else
	echo "Directory ~/OpenCV already exists, remove the directory before proceding"
	echo "Closing"
	exit
fi


# DOWNLOADING OpenCV Source
mkdir ~/OpenCV
cd ~/OpenCV

VERSION=4.4.0
git clone https://github.com/opencv/opencv.git -b $VERSION --depth 1
git clone https://github.com/opencv/opencv_contrib.git -b $VERSION --depth 1


# BUILDING
cd opencv
mkdir build
pkg-config --cflags --libs gstreamer-1.0
cd build
cmake -D CMAKE_BUILD_TYPE=RELEASE \
-D CMAKE_INSTALL_PREFIX=/usr/local \
-D OPENCV_GENERATE_PKGCONFIG=ON \
-D BUILD_EXAMPLES=OFF \
-D INSTALL_PYTHON_EXAMPLES=OFF \
-D INSTALL_C_EXAMPLES=OFF \
-D PYTHON_EXECUTABLE=$(which python2) \
-D BUILD_opencv_python2=OFF \
-D PYTHON3_EXECUTABLE=$(which python3) \
-D PYTHON3_INCLUDE_DIR=$(python3 -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())") \
-D PYTHON3_PACKAGES_PATH=$(python3 -c "from distutils.sysconfig import get_python_lib; print(get_python_lib())") \
-D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules/ \
-D WITH_GSTREAMER=ON \
..

make -j8

# INSTALLING
sudo make install
sudo ldconfig
