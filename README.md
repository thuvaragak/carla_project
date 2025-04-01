# carla_project
Developing an Enhanced Perception system in CARLA using ROS 2, integrating LiDAR, radar, and camera data for sensor fusion, real-time object tracking, and visualization using Open3D.

## Overview
This project implements an **Enhanced Perception system** in the **CARLA simulator** using **ROS 2**, integrating sensor data from **LiDAR, radar, and cameras** for improved object tracking and environment perception.

## Features
- **Sensor Data Acquisition**: ROS 2 nodes for extracting LiDAR, radar, and camera data from CARLA.
- **Visualization**: Open3D-based visualization for fused sensor data.

## Technologies Used
- **CARLA** (Autonomous Driving Simulator)
- **ROS 2** (Robotic Operating System)
- **PCL** (Point Cloud Library)
- **Open3D** (Visualization & 3D Processing)
- **Python & C++** (For ROS nodes and sensor processing)


## Running the System
1. **Launch CARLA**:
   ```bash
   ./CarlaUE4.sh -prefernvidia -quality-level=Low -carla-server -benchmark -fps=15 -windowed -ResX=800 -ResY=600
   ```
2. **Run ROS 2 Nodes**:
   ```bash
   ros2 run carla_project vehicle
   ```


## Future Enhancements
- Implement **Deep Learning-based Object Detection**.
- Integrate **SLAM** for mapping.
- Improve Sensor Fusion.
- **Sensor Calibration**: Intrinsic and extrinsic calibration for accurate sensor alignment.
- **Feature Extraction**: Processing raw sensor data for object recognition.
- **Sensor Fusion**: Combining multiple sensor inputs for enhanced perception.
- **Object Tracking**: Real-time object detection and tracking.

---
**Author**: Thuvaraga K
