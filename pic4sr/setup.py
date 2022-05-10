from setuptools import setup
import os 
from glob import *

package_name = 'pic4sr'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml'))
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='mauromartini',
    maintainer_email='mauro.martini@polito.it',
    description='SUPER RESOLUTION image processing for multi-platform robotic remote navigation',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
        'pic4sr_realsense = pic4sr.pic4sr_realsense:main',
        'pic4sr_cam = pic4sr.pic4sr_cam:main',
	'pic4sr_ground = pic4sr.pic4sr_ground:main',
        ],
    },
)
