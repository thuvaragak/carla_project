# carla_project
Developing an ADAS System in CARLA Simulator.

## Overview
This project implements a **Ego Vehicle Sensor System** in the **CARLA simulator** using **ROS 2**, integrating sensor data from **LiDAR, radar, and cameras** and **ADAS: Vehicle Detection Using LiDAR Data through RANSAC Segmentation and Euclidean Clustering.**

## Files
- **vehicle.py**: ROS 2 nodes for extracting LiDAR, radar, and camera data from CARLA. Open3D-based visualization for fused sensor data.
- **adas_vehicle_detection.py**: ADAS: Vehicle Detection Using LiDAR Data through RANSAC Segmentation and Euclidean Clustering.
     - Vehicle setup, Sensor_listen:  To launch CARLA & Spawn vehicles in the CARLA world.  To attach sensors(Lidar, Camera) to the ego vehicle and get sensor data from the CARLA environment
     -   Lidar_callback: Executing vehicle detection algorithms in Lidar data
     -   Sensor_visualization: To visualize sensor output
     -   Segment_plane_ransac: To split the lidar point cloud into the Road plane and the Obstacle plane. The road plane will be visualized as green points, and obstacles as red points
     -   Clustering_euclidean_o3d: Applying the Euclidean clustering algorithm to detect vehicles or any obstacles in the plane.
     -   Filter_cloud_from_numpy: Filtering noise data from raw lidar data.
- **liosam_carla.py**: ROS 2 nodes for setting up LiDAR, IMU, GNSS, camera and other sensors with TF2 transformations. Process and publish Lidar, IMU & GNSS data from Lio-SAM (https://github.com/TixiaoShan/LIO-SAM/tree/ros2). 

## Technologies Used
- **CARLA** (Autonomous Driving Simulator)
- **ROS 2** (Robotic Operating System)
- **PCL** (Point Cloud Library)
- **Open3D** (Visualization & 3D Processing)
- **Python & C++** (For ROS nodes and sensor processing)


## Executing code to launch an ego vehicle with sensors and visualize it
1. **Launch CARLA**:
   ```bash
   ./CarlaUE4.sh -prefernvidia -quality-level=Low -carla-server -benchmark -fps=15 -windowed -ResX=800 -ResY=600
   ```
2. **Run ROS 2 Nodes**:
   ```bash
   ros2 run carla_project vehicle
   ```
## Executing code to launch ADAS Vehicle detection pipeline
1. **Launch CARLA**:
   ```bash
   ./CarlaUE4.sh -prefernvidia -quality-level=Low -carla-server -benchmark -fps=15 -windowed -ResX=800 -ResY=600
   ```
2. **Run ROS 2 Nodes**:
   ```bash
   ros2 run carla_project adas_vehicle_detection
   ```
---
**Author**: Thuvaraga K
