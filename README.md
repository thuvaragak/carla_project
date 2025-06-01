# carla_project
Developing an Enhanced Perception system in CARLA using ROS 2, integrating LiDAR, radar, and camera data for sensor fusion, real-time object tracking, and visualization using Open3D.

## Overview
This project implements an **Enhanced Perception system** in the **CARLA simulator** using **ROS 2**, integrating sensor data from **LiDAR, radar, and cameras** for improved object tracking and environment perception.

## Files
- **vehicle.py**: ROS 2 nodes for extracting LiDAR, radar, and camera data from CARLA.Open3D-based visualization for fused sensor data.
- **adas_vehicle_detection.py**:
     - Vehicle setup, Sensor_listen:  To launch CARLA & Spawn vehicles in the CARLA world.  To attach sensors(Lidar, Camera) to the ego vehicle and get sensor data from the CARLA environment
     -   Lidar_callback: Executing vehicle detection algorithms in Lidar data
     -   Sensor_visualization: To visualize sensor output
     -   Segment_plane_ransac: To split the lidar point cloud into the Road plane and the Obstacle plane. The road plane will be visualized as green points, and obstacles as red points
     -   Clustering_euclidean_o3d: Applying the Euclidean clustering algorithm to detect vehicles or any obstacles in the plane.
     -   Filter_cloud_from_numpy: Filtering noise data from raw lidar data.


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
