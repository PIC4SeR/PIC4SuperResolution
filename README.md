> Project: "PIC4SuperResolution"

> Owner: "PIC4SeR" 

> Date: "2022:05" 

---

# Title
PIC4SuperResolution
## Description of the project
The project aims at developing real-time image upscaling with Deep Learning at the edge to support visual remote control of UGVs and UAVs in exploration missions. The system allows to speed up and improve control on teleoperated robotic operation:
1) Images captured by the robot on-board camera (Intel RealSense 435i) are transmitted with a minimal resolution at high frequnency. 
2) An SRGAN model is trained to provide a x4 upscaling of the images received at the ground station / visual controller.
3) The operator can guide the robot with an optimized visual stream. 

Two ROS nodes:
  - pic4sr_realsense: run the camera acquisition and publishing process on the robot
  - pic4sr_ground: receive the images and perform the upscaling with the SR model

Config params:
  - model path: specify model path
  - sensor: rgb images and depth images
  - image width: input resolution of the model
  - image height: input resolution of the model
  - device: run on CPU (TFlite) or on Coral EdgeTPU

## Installation procedure
- Install RealSense D435i SDK + pyrealsense2
- Install ROS2
- Install TensorFlow2 (2.6 or greater)
- Install Coral EdgeTPU requirements from the website https://coral.ai/docs/accelerator/get-started/

## User Guide
...

P.S. Compile requirements.txt file if needed

More detailed information about markdown style for README.md file [HERE](https://github.com/adam-p/markdown-here/wiki/Markdown-Cheatsheet)
