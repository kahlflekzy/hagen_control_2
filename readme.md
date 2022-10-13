Create workspace
Create an `src` sub-folder
Run
```
cd src && git clone https://github.com/GPrathap/autonomous_mobile_robots.git
```
Run
```
cd .. && rosdep install --from-paths src --ignore-src -r -y
```
Then
```
colcon build
```

If `colcon` is not installed, run
```
sudo apt install python3-colcon-common-extensions
```
If catkin has been installed run
```
python -m pip install --upgrade pip && python -m pip install catkin_pkg
```
In another terminal,
> Highly Important
```
 . /usr/share/gazebo/setup.sh
 . install/setup.bash
```

Confirm all is up by running
```
ros2 pkg list | grep hagen_gazebo
```
Then run
```
ros2 launch hagen_gazebo hagen.launch.py world:=hagen_city.world
```

Added an executable to the Hagen_control Package

Create a package using
```
ros2 pkg create --build-type ament_python --license BSD-2.0 --description "A package for my tutorials on controlling the hagen wheeled robot." --dependencies rclpy std_msgs sensor_msgs geometry_msgs nav_msgs --node-name controller hagen_control_2
```

in the workspace, install missing packages using
```
rosdep install -i --from-path src --rosdistro $ROS_DISTRO -y
```
 sudo apt install $ROS_DISTRO-

Then build using
```
colcon build --symlink-install --packages-select hagen_control_2
```

> IMPORTANT: To see the `ENV`
> ```
> printenv | grep -i ROS
> ```
> 
> 

## Add ROS support to pycharm
1. Source the ROS underlay from a terminal. 
2. Run pycharm from the terminal, like so `pycharm-community`
3. Add the ros packages to project structure (Project Structure->Add Content Root)
`/opt/ros/$ROS_DISTRO/lib/python3.8/site-packages`

to install `gazebo` run
```
sudo apt install gazebo
```
to install gazebo ros packages, run
```
sudo apt install ros-$ROS_DISTRO-gazebo-ros-pkgs
```


Added bin path to paths so pycharm can be ran anywhere
run 
```
gedit ~/.bashrc
```
and type at the bottom
```
PATH="~/Downloads/pycharm-community-2022.2.2/bin:$PATH$"
```

convert `install-script` to `install_scripts` and `script-dir` to `script_dir` in `setup.cfg`


In 

```
ros2 pkg create --license BSD-2.0 --description "A package for hagen control" --dependencies rclpy std_msgs sensor_msgs geometry_msgs nav_msgs --build-type ament_python --node-name controller control
```
### REFERENCES 
1. [Time](https://github.com/mikeferguson/ros2_cookbook/blob/main/rclpy/time.md)