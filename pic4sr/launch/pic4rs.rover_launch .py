import os
from ament_index_python.packages import get_package_share_directory
import launch
from launch_ros.actions import Node
from launch.actions import ExecuteProcess

def generate_launch_description():

    config = os.path.join(
        get_package_share_directory('pic4sr'),
        'config',
        'params.yaml')

    pic4sr_realsense = Node(
        package='pic4sr',
        node_executable='pic4sr_realsense',
        node_name='pic4sr_realsense',
        prefix=['stdbuf -o L'],
        output='screen',
        parameters=[config])
    
    return launch.LaunchDescription([
        pic4sr_realsense
    ])
