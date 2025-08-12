# bevfusion_ros_inference
ROS inference node for BEVFusion

# Installation

1. Set Up the Workspace

mkdir -p ~/bevfusion_ws/src
cd ~/bevfusion_ws
catkin_make clean
catkin_make --cmake-args -DCMAKE_POLICY_VERSION_MINIMUM=3.5
source /home/nvidia/bevfusion_ws/devel/setup.bash (replace with your local path)

2. Create the ROS Package
3. 
cd ~/bevfusion_ws/src
catkin_create_pkg bevfusion_onnx sensor_msgs roscpp rospy std_msgs

4. Organize the Package Structure

Create the scripts directory and place your bevfusion_onnx_node.py script in it:
chmod +x ~/bevfusion_ws/src/bevfusion_onnx/scripts/bevfusion_onnx_node.py
Create a launch directory for launch files:
mkdir -p ~/bevfusion_ws/src/bevfusion_onnx/launch

6. Edit package.xml

7. Edit CMakeLists.txt

8. Create & Edit bevfusion_onnx.launch

How to use:

cd bevfusion_ws/src/bevfusion_onnx
source /home/nvidia/bevfusion_ws/devel/setup.bash'

Terminal 1: roscore

Terminal 2: rosrun bevfusion_onnx publish_data.py 

Terminal 3: rosrun bevfusion_onnx bevfusion_onnx_node.py


The folder structure:
'''
bevfusion_ws/src/bevfusion_onnx/
├── CMakeLists.txt
├── package.xml
├── scripts/
│   └── bevfusion_onnx_node.py
    └── publish_data.py
├── launch/
│   └── bevfusion_onnx.launch
├── msg/
'''
