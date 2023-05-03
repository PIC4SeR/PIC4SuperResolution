> Project: "PIC4SuperResolution"

> Owner: "PIC4SeR" 

> Date: "2022:05" 

---

# PIC4SuperResolution
This is the second repository of the work "Generative Adversarial Super-Resolution at the Edge with Knowledge Distillation". The repo contains the ROS 2 package used to test the proposed Super-Resolution model at the edge in robotic applications.
## Description of the project
The project aims at developing real-time image upscaling with Deep Learning at the edge to support visual remote control of UGVs and UAVs in exploration missions. The system allows to speed up and improve control on teleoperated robotic operation:
1) Images captured by the robot on-board camera (Intel RealSense 435i) are transmitted with a minimal resolution at high frequnency. 
2) An SRGAN model is trained to provide a x4 upscaling of the images received at the ground station / visual controller.
3) The operator can guide the robot with an optimized visual stream. 

Two ROS nodes:
  - pic4sr_realsense or pic4sr_cam: run the realsense or webcam acquisition and publishing process on the robot
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

## References
Angarano, S., Salvetti, F., Martini, M., & Chiaberge, M. (2022). Generative Adversarial Super-Resolution at the Edge with Knowledge Distillation. arXiv preprint arXiv:2209.03355.

	@article{angarano2022generative,
	  title={Generative Adversarial Super-Resolution at the Edge with Knowledge Distillation},
	  author={Angarano, Simone and Salvetti, Francesco and Martini, Mauro and Chiaberge, Marcello},
	  journal={arXiv preprint arXiv:2209.03355},
	  year={2022}
	}

