import rclpy
from rclpy.node import Node
import carla 
import math 
import random 
import time 
import numpy as np
import cv2
from cv_bridge import CvBridge
import open3d as o3d
from matplotlib import cm

from sensor_msgs.msg import CameraInfo, Image, PointCloud2, PointField, NavSatFix, Imu
from std_msgs.msg import Header
from nav_msgs.msg import Odometry

import struct
import ctypes
import ros_compatibility as roscomp
from abc import abstractmethod

from geometry_msgs.msg import TransformStamped, Quaternion
from tf2_ros import StaticTransformBroadcaster

import mgrs
import utm

# Auxilliary code for colormaps and axes
VIRIDIS = np.array(cm.get_cmap('plasma').colors)
VID_RANGE = np.linspace(0.0, 1.0, VIRIDIS.shape[0])
COOL_RANGE = np.linspace(0.0, 1.0, VIRIDIS.shape[0])
COOL = np.array(cm.get_cmap('winter')(COOL_RANGE))
COOL = COOL[:,:3]

_DATATYPES = {}
_DATATYPES[PointField.INT8] = ('b', 1)
_DATATYPES[PointField.UINT8] = ('B', 1)
_DATATYPES[PointField.INT16] = ('h', 2)
_DATATYPES[PointField.UINT16] = ('H', 2)
_DATATYPES[PointField.INT32] = ('i', 4)
_DATATYPES[PointField.UINT32] = ('I', 4)
_DATATYPES[PointField.FLOAT32] = ('f', 4)
_DATATYPES[PointField.FLOAT64] = ('d', 8)

def _get_struct_fmt(is_bigendian, fields, field_names=None):
    fmt = '>' if is_bigendian else '<'

    offset = 0
    for field in (f for f in sorted(fields, key=lambda f: f.offset)
                  if field_names is None or f.name in field_names):
        if offset < field.offset:
            fmt += 'x' * (field.offset - offset)
            offset = field.offset
        if field.datatype not in _DATATYPES:
            print('Skipping unknown PointField datatype [{}]' % field.datatype, file=sys.stderr)
        else:
            datatype_fmt, datatype_length = _DATATYPES[field.datatype]
            fmt += field.count * datatype_fmt
            offset += field.count * datatype_length

    return fmt

def create_cloud(header, fields, points):
    cloud_struct = struct.Struct(_get_struct_fmt(False, fields))

    buff = ctypes.create_string_buffer(cloud_struct.size * len(points))

    point_step, pack_into = cloud_struct.size, cloud_struct.pack_into
    offset = 0
    for p in points:
        pack_into(buff, offset, *p)
        offset += point_step

    return PointCloud2(header=header,
                       height=1,
                       width=len(points),
                       is_dense=False,
                       is_bigendian=False,
                       fields=fields,
                       point_step=cloud_struct.size,
                       row_step=cloud_struct.size * len(points),
                       data=buff.raw)


def add_open3d_axis(vis):
    """Add a small 3D axis on Open3D Visualizer"""
    axis = o3d.geometry.LineSet()
    axis.points = o3d.utility.Vector3dVector(np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]]))
    axis.lines = o3d.utility.Vector2iVector(np.array([
        [0, 1],
        [0, 2],
        [0, 3]]))
    axis.colors = o3d.utility.Vector3dVector(np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]]))
    vis.add_geometry(axis)

class LiosamCarla(Node):
    def __init__(self):
        super().__init__('liosam_carla')

        #CARLA INIT
        self.client = carla.Client('localhost', 2000) 
        self.world = self.client.get_world()
        self.bp_lib = self.world.get_blueprint_library() 
        self.br = CvBridge()
        self.vehicle = None
        self.lidar = None
        self.lidar_rear = None
        self.gnss = None
        self.imu = None
        self.collision = None
        self.lane_invasion = None
        self.odom = None
        self.radar = None
        self.camera = None

        self.mgrs_converter = mgrs.MGRS()

        # Set up base_link transform
        self.tf_broadcaster = StaticTransformBroadcaster(self)
        self.tf_broadcaster_lidar = StaticTransformBroadcaster(self)
        self.tf_broadcaster_lidar_rear = StaticTransformBroadcaster(self)
        self.tf_broadcaster_camera = StaticTransformBroadcaster(self)
        self.tf_broadcaster_gnss = StaticTransformBroadcaster(self)
        self.tf_broadcaster_imu = StaticTransformBroadcaster(self)
        self.tf_broadcaster_coll = StaticTransformBroadcaster(self)
        self.tf_broadcaster_lane = StaticTransformBroadcaster(self)
        self.tf_broadcaster_odom = StaticTransformBroadcaster(self)

        self.img_pub = self.create_publisher(Image, 'carla/img', 10)
        self.lidar_pub = self.create_publisher(PointCloud2, '/points', 10)
        self.radar_pub = self.create_publisher(PointCloud2, 'carla/radar', 10)
        self.info_pub = self.create_publisher(CameraInfo, 'carla/camera_info', 10)
        self.imu_pub = self.create_publisher(Imu, '/imu/data', 10)
        self.gnss_pub = self.create_publisher(Odometry, '/odometry/gpsz', 10)

        # Add auxilliary data structures
        self.camera_data = None
        self.carla_actor = None 
        self.camera_info = None
        self.gnss_data = None
        self.imu_data = None

        self.point_list = o3d.geometry.PointCloud()
        self.radar_list = o3d.geometry.PointCloud()

        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(
            window_name='Carla Lidar',
            width=960,
            height=540,
            left=480,
            top=270)

        self.vis.get_render_option().background_color = [0.05, 0.05, 0.05]
        self.vis.get_render_option().point_size = 1
        self.vis.get_render_option().show_coordinate_frame = True
        add_open3d_axis(self.vis)

        # Vehicle Setup
        self.vehicle_setup()
        
        self.sensor_listen()
        self.sensor_visualization()
        self.vehicle_destroy()
        self._control = carla.VehicleControl()

        # self.create_timer(0.2, self.timer_callback)

    def _build_camera_info(self, image_w, image_h, fov):
        camera_info = CameraInfo()
        # store info without header
        camera_info.header = self.get_msg_header()
        camera_info.width = image_w #int(self.carla_actor.attributes['image_size_x'])
        camera_info.height = image_h #int(self.carla_actor.attributes['image_size_y'])
        camera_info.distortion_model = 'plumb_bob'
        cx = camera_info.width / 2.0
        cy = camera_info.height / 2.0
        fx = camera_info.width / (2.0 * math.tan(float(fov) * math.pi / 360.0))
        fy = fx
        camera_info.k = [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]
        camera_info.d = [0.0, 0.0, 0.0, 0.0, 0.0]
        camera_info.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        camera_info.p = [fx, 0.0, cx, 0.0, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0]
        self.camera_info = camera_info

    def radar_data_updated(self, carla_radar_measurement):
        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='Range', offset=12, datatype=PointField.FLOAT32, count=1),
            PointField(name='Velocity', offset=16, datatype=PointField.FLOAT32, count=1),
            PointField(name='AzimuthAngle', offset=20, datatype=PointField.FLOAT32, count=1),
            PointField(name='ElevationAngle', offset=28, datatype=PointField.FLOAT32, count=1)]
        points = []

        for detection in carla_radar_measurement:
            points.append([detection.depth * np.cos(detection.azimuth) * np.cos(-detection.altitude),
                           detection.depth * np.sin(-detection.azimuth) *
                           np.cos(detection.altitude),
                           detection.depth * np.sin(detection.altitude),
                           detection.depth, detection.velocity, detection.azimuth, detection.altitude])
        radar_msg = create_cloud(self.get_msg_header(
            timestamp=carla_radar_measurement.timestamp), fields, points)
        self.radar_pub.publish(radar_msg)

    def radar_callback(self, data, point_list):
        self.radar_data_updated(data)
        radar_data = np.zeros((len(data), 4))
        
        for i, detection in enumerate(data):
            x = detection.depth * math.cos(detection.altitude) * math.cos(detection.azimuth)
            y = detection.depth * math.cos(detection.altitude) * math.sin(detection.azimuth)
            z = detection.depth * math.sin(detection.altitude)
            
            radar_data[i, :] = [x, y, z, detection.velocity]
            
        intensity = np.abs(radar_data[:, -1])
        intensity_col = 1.0 - np.log(intensity) / np.log(np.exp(-0.004 * 100))
        int_color = np.c_[
            np.interp(intensity_col, COOL_RANGE, COOL[:, 0]),
            np.interp(intensity_col, COOL_RANGE, COOL[:, 1]),
            np.interp(intensity_col, COOL_RANGE, COOL[:, 2])]
        
        points = radar_data[:, :-1]
        points[:, :1] = -points[:, :1]
        point_list.points = o3d.utility.Vector3dVector(points)
        point_list.colors = o3d.utility.Vector3dVector(int_color)

    def get_msg_header(self, timestamp=None):
        header = Header()
        if not timestamp:
            header.stamp = self.get_clock().now().to_msg()
        else:
            header.stamp = roscomp.ros_timestamp(sec=timestamp, from_sec=True)
        return header
    
    def camera_callback(self, image, data_dict):
        data_dict['image'] = np.reshape(np.copy(image.raw_data), (image.height, image.width, 4)) 
        img = self.get_ros_image(image)
        cam_info = self.camera_info
        cam_info.header = img.header

        self.img_pub.publish(img)
        self.info_pub.publish(cam_info)

    def gnss_callback(self, data, data_dict):
        print("GNSS", data)
        # mgrs_str = self.mgrs_converter.toMGRS(data.latitude, data.longitude)
        utm_result = utm.from_latlon(data.latitude, data.longitude)
        easting = utm_result[0]
        northing = utm_result[1]

        odom = Odometry()
        odom.header.stamp = self.get_clock().now().to_msg()
        odom.header.frame_id = "map"
        odom.child_frame_id = "navsat_link"

        odom.pose.pose.position.x = easting
        odom.pose.pose.position.y = northing
        odom.pose.pose.position.z = data.altitude

        odom.pose.pose.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
        self.gnss_pub.publish(odom)

        # navsatfix_msg = NavSatFix()
        # navsatfix_msg.header = self.get_msg_header(timestamp=data.timestamp)
        # navsatfix_msg.latitude = data.latitude
        # navsatfix_msg.longitude = data.longitude
        # navsatfix_msg.altitude = data.altitude
        # data_dict['gnss'] = [data.latitude, data.longitude]

    def imu_callback(self, data, data_dict):
        imu_msg = Imu() 
        imu_msg.header.stamp = self.get_clock().now().to_msg()
        imu_msg.header.frame_id = 'imu_link'
        imu_msg.linear_acceleration.x = data.accelerometer.x
        imu_msg.linear_acceleration.y = data.accelerometer.y
        imu_msg.linear_acceleration.z = data.accelerometer.z
        imu_msg.angular_velocity.x = data.gyroscope.x
        imu_msg.angular_velocity.y = data.gyroscope.y
        imu_msg.angular_velocity.z = data.gyroscope.z
        self.imu_pub.publish(imu_msg)

    def lidar_callback(self, point_cloud, point_list):
        """Prepares a point cloud with intensity colors ready to be consumed by Open3D"""
        data = np.copy(np.frombuffer(point_cloud.raw_data, dtype=np.dtype('f4')))
        data = np.reshape(data, (int(data.shape[0] / 4), 4))

        # Isolate the intensity and compute a color for it
        intensity = data[:, -1]
        intensity_col = 1.0 - np.log(intensity) / np.log(np.exp(-0.004 * 100))
        int_color = np.c_[
            np.interp(intensity_col, VID_RANGE, VIRIDIS[:, 0]),
            np.interp(intensity_col, VID_RANGE, VIRIDIS[:, 1]),
            np.interp(intensity_col, VID_RANGE, VIRIDIS[:, 2])]

        points = data[:, :-1]

        points[:, :1] = -points[:, :1]

        point_list.points = o3d.utility.Vector3dVector(points)
        point_list.colors = o3d.utility.Vector3dVector(int_color)

        # Convert CarlaLidarMeasuremt to ROS2 pointcloud2
        header = self.get_msg_header(timestamp=point_cloud.timestamp)
        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='intensity', offset=12, datatype=PointField.FLOAT32, count=1)
        ]

        lidar_data = np.fromstring(
            bytes(point_cloud.raw_data), dtype=np.float32)
        lidar_data = np.reshape(
            lidar_data, (int(lidar_data.shape[0] / 4), 4))
        # we take the opposite of y axis
        # (as lidar point are express in left handed coordinate system, and ros need right handed)
        lidar_data[:, 1] *= -1

        header.frame_id = "lidar_link"
        point_cloud_msg = create_cloud(header, fields, lidar_data)
        self.lidar_pub.publish(point_cloud_msg)

    def publish_base_link_tf(self):
        static_tf = TransformStamped()
        static_tf.header.stamp = self.get_clock().now().to_msg()
        static_tf.header.frame_id = 'base_link'  # CARLA's default frame
        static_tf.child_frame_id = 'chassis_link'     # ROS standard frame
        
        # Adjust these values if base_link needs an offset (e.g., rear axle)
        static_tf.transform.translation.x = 0.0
        static_tf.transform.translation.y = 0.0
        static_tf.transform.translation.z = 0.0
        static_tf.transform.rotation.w = 1.0  # No rotation
        self.tf_broadcaster.sendTransform(static_tf)

        # LIDAR Front
        lidar_tf = TransformStamped()
        lidar_tf.header.stamp = self.get_clock().now().to_msg()
        lidar_tf.header.frame_id = 'base_link'  # Parent frame
        lidar_tf.child_frame_id = 'lidar_link'  # Front LiDAR frame
        lidar_tf.transform.translation.x = 1.5  # Forward # Must match the CARLA spawn transform!
        lidar_tf.transform.translation.y = 0.0   # Right
        lidar_tf.transform.translation.z = 2.1   # Up
        lidar_tf.transform.rotation.w = 1.0
        lidar_tf.transform.rotation.x = 0.0
        lidar_tf.transform.rotation.y = 0.0
        lidar_tf.transform.rotation.z = 0.0
        self.tf_broadcaster_lidar.sendTransform(lidar_tf)

        # LIDAR Rear
        # lidar_rear_tf = TransformStamped()
        # lidar_rear_tf.header.stamp = self.get_clock().now().to_msg()
        # lidar_rear_tf.header.frame_id = 'base_link'  # Parent frame
        # lidar_rear_tf.child_frame_id = 'lidar_rear'  # Rear LiDAR frame
        # lidar_rear_tf.transform.translation.x = -1.5  # Forward # Must match the CARLA spawn transform!
        # lidar_rear_tf.transform.translation.y = 0.0   # Right
        # lidar_rear_tf.transform.translation.z = 2.1   # Up
        # lidar_rear_tf.transform.rotation.w = 1.0
        # lidar_rear_tf.transform.rotation.x = 0.0
        # lidar_rear_tf.transform.rotation.y = 0.0
        # lidar_rear_tf.transform.rotation.z = 0.0
        # self.tf_broadcaster_lidar_rear.sendTransform(lidar_rear_tf)

        # CAMERA
        camera_tf = TransformStamped()
        camera_tf.header.stamp = self.get_clock().now().to_msg()
        camera_tf.header.frame_id = 'base_link'  # Parent frame
        camera_tf.child_frame_id = 'camera_front_link'  # Camera frame
        camera_tf.transform.translation.x = 1.8  # Forward
        camera_tf.transform.translation.y = 0.25   # Right
        camera_tf.transform.translation.z = 1.65   # Up
        camera_tf.transform.rotation.w = 1.0
        camera_tf.transform.rotation.x = 0.0
        camera_tf.transform.rotation.y = 0.0
        camera_tf.transform.rotation.z = 0.0
        self.tf_broadcaster_camera.sendTransform(camera_tf)

        # GNSS
        gnss_tf = TransformStamped()
        gnss_tf.header.stamp = self.get_clock().now().to_msg()
        gnss_tf.header.frame_id = 'chassis_link'  # Parent frame
        gnss_tf.child_frame_id = 'navsat_link'  # Camera frame
        gnss_tf.transform.translation.x = 0.0  # Forward
        gnss_tf.transform.translation.y = 0.0   # Right
        gnss_tf.transform.translation.z = 2.1   # Up
        gnss_tf.transform.rotation.w = 1.0
        gnss_tf.transform.rotation.x = 0.0
        gnss_tf.transform.rotation.y = 0.0
        gnss_tf.transform.rotation.z = 0.0
        self.tf_broadcaster_gnss.sendTransform(gnss_tf)

        # IMU
        imu_tf = TransformStamped()
        imu_tf.header.stamp = self.get_clock().now().to_msg()
        imu_tf.header.frame_id = 'chassis_link'  # Parent frame
        imu_tf.child_frame_id = 'imu_link'  # Camera frame
        imu_tf.transform.translation.x = 0.1  # Forward
        imu_tf.transform.translation.y = 0.0   # Right
        imu_tf.transform.translation.z = 2.1   # Up
        imu_tf.transform.rotation.w = 1.0
        imu_tf.transform.rotation.x = 0.0
        imu_tf.transform.rotation.y = 0.0
        imu_tf.transform.rotation.z = 0.0
        self.tf_broadcaster_imu.sendTransform(imu_tf)

        # Collision
        coll_tf = TransformStamped()
        coll_tf.header.stamp = self.get_clock().now().to_msg()
        coll_tf.header.frame_id = 'base_link'  # Parent frame
        coll_tf.child_frame_id = 'collision_link'  # Camera frame
        coll_tf.transform.translation.x = 0.0 # Forward
        coll_tf.transform.translation.y = 0.0   # Right
        coll_tf.transform.translation.z = 0.0   # Up
        self.tf_broadcaster_coll.sendTransform(coll_tf)

        # Lane Invasion
        lane_tf = TransformStamped()
        lane_tf.header.stamp = self.get_clock().now().to_msg()
        lane_tf.header.frame_id = 'base_link'  # Parent frame
        lane_tf.child_frame_id = 'lane_link'  # Camera frame
        lane_tf.transform.translation.x = 0.0  # Forward
        lane_tf.transform.translation.y = 0.0   # Right
        lane_tf.transform.translation.z = 0.0   # Up
        self.tf_broadcaster_lane.sendTransform(lane_tf)

        # ODOM
        odom_tf = TransformStamped()
        odom_tf.header.stamp = self.get_clock().now().to_msg()
        odom_tf.header.frame_id = 'odom'  # Parent frame
        odom_tf.child_frame_id = 'base_link'  # Camera frame
        odom_tf.transform.translation.x = 0.0  # Forward
        odom_tf.transform.translation.y = 0.0   # Right
        odom_tf.transform.translation.z = 0.0   # Up
        odom_tf.transform.rotation.w = 1.0
        odom_tf.transform.rotation.x = 0.0
        odom_tf.transform.rotation.y = 0.0
        odom_tf.transform.rotation.z = 0.0
        self.tf_broadcaster_odom.sendTransform(odom_tf)

    def vehicle_setup(self):
        self.spawn_points = self.world.get_map().get_spawn_points() 
        self.vehicle_bp = self.bp_lib.find('vehicle.audi.a2') 
        self.vehicle_bp.set_attribute('role_name', 'hero')
        self.vehicle = self.world.try_spawn_actor(self.vehicle_bp, random.choice(self.spawn_points))
        
        # Move spectator behind vehicle to view
        spectator = self.world.get_spectator() 
        transform = carla.Transform(self.vehicle.get_transform().transform(carla.Location(x=-4,z=2.5)),self.vehicle.get_transform().rotation) 
        spectator.set_transform(transform)

        # Add traffic and set in motion with Traffic Manager
        for i in range(100): 
            self.vehicle_bp = random.choice(self.bp_lib.filter('vehicle')) 
            npc = self.world.try_spawn_actor(self.vehicle_bp, random.choice(self.spawn_points))    
        for v in self.world.get_actors().filter('*vehicle*'): 
            v.set_autopilot(True) 
        

        # LiDAR Front
        lidar_bp = self.bp_lib.find('sensor.lidar.ray_cast') 
        lidar_bp.set_attribute('range', '100.0')
        lidar_bp.set_attribute('noise_stddev', '0.0')
        lidar_bp.set_attribute('upper_fov', '3.0')
        lidar_bp.set_attribute('lower_fov', '-27.0')
        lidar_bp.set_attribute('channels', '64.0')
        lidar_bp.set_attribute('rotation_frequency', '20.0')
        lidar_bp.set_attribute('points_per_second', '480000')
        loc = {"x": 0.0, "y": 0.0, "z": 2.100, "roll": 0.0, "pitch": 0.0, "yaw": 0.0} 
        lidar_init_trans = carla.Transform(
                                carla.Location(x=loc["x"], y=loc["y"], z=loc["z"]),
                                carla.Rotation(roll=loc["roll"], pitch=loc["pitch"], yaw=loc["yaw"]))
        self.lidar = self.world.spawn_actor(lidar_bp, lidar_init_trans, attach_to=self.vehicle)

        # LiDAR Rear
        # lidar_rear_bp = self.bp_lib.find('sensor.lidar.ray_cast') 
        # lidar_rear_bp.set_attribute('range', '50.0')
        # lidar_rear_bp.set_attribute('noise_stddev', '0.0')
        # lidar_rear_bp.set_attribute('upper_fov', '3.0')
        # lidar_rear_bp.set_attribute('lower_fov', '-27.0')
        # lidar_rear_bp.set_attribute('channels', '32.0')
        # lidar_rear_bp.set_attribute('rotation_frequency', '20.0')
        # lidar_rear_bp.set_attribute('points_per_second', '96000')
        # rear_loc = {"x": -1.5, "y": 0.0, "z": 2.100, "roll": 0.0, "pitch": 0.0, "yaw": 0.0} 
        # lidar_init_trans_rear = carla.Transform(
        #                         carla.Location(x=rear_loc["x"], y=rear_loc["y"], z=rear_loc["z"]),
        #                         carla.Rotation(roll=rear_loc["roll"], pitch=rear_loc["pitch"], yaw=rear_loc["yaw"]))
        # self.lidar_rear = self.world.spawn_actor(lidar_rear_bp, lidar_init_trans_rear, attach_to=self.vehicle)

        # Front CAMERA
        camera_bp = self.bp_lib.find('sensor.camera.rgb') 
        camera_bp.set_attribute('image_size_x', '800')
        camera_bp.set_attribute('image_size_y', '600')
        camera_bp.set_attribute('fov', '90')

        # camera_init_trans = carla.Transform(carla.Location(z=1.5,x=1.1), carla.Rotation())
        cam_loc = {"x": 1.80, "y": 0.25, "z": 1.65, "roll": 0.0, "pitch": 0.0, "yaw": 0.0}
        camera_init_trans = carla.Transform(
                                carla.Location(x=cam_loc["x"], y=cam_loc["y"], z=cam_loc["z"]),
                                carla.Rotation(roll=cam_loc["roll"], pitch=cam_loc["pitch"], yaw=cam_loc["yaw"]))
        self.camera = self.world.spawn_actor(camera_bp, camera_init_trans, attach_to=self.vehicle)
        
        image_w = camera_bp.get_attribute("image_size_x").as_int()
        image_h = camera_bp.get_attribute("image_size_y").as_int()
        fov = camera_bp.get_attribute("fov")
        self._build_camera_info(image_w, image_h, fov)
        self.camera_data = {'image': np.zeros((image_h, image_w, 4))} 

        # GNSS
        gnss_bp = self.bp_lib.find('sensor.other.gnss')
        gnss_loc = {"x": 0.0, "y": 0.0, "z": 2.1, "roll": 0.0, "pitch": 0.0, "yaw": 0.0}
        gnss_bp.set_attribute('noise_alt_bias', '0.0')       # Altitude noise bias
        gnss_bp.set_attribute('noise_alt_stddev', '0.0')     # Altitude noise stddev
        gnss_bp.set_attribute('noise_lat_bias', '0.0')       # Latitude noise bias
        gnss_bp.set_attribute('noise_lat_stddev', '0.0')     # Latitude noise stddev
        gnss_bp.set_attribute('noise_lon_bias', '0.0')       # Longitude noise bias
        gnss_bp.set_attribute('noise_lon_stddev', '0.0')     # Longitude noise stddev
        gnss_init_trans = carla.Transform(
                                carla.Location(x=gnss_loc["x"], y=gnss_loc["y"], z=gnss_loc["z"]),
                                carla.Rotation(roll=gnss_loc["roll"], pitch=gnss_loc["pitch"], yaw=gnss_loc["yaw"]))
        self.gnss = self.world.spawn_actor(gnss_bp, gnss_init_trans, attach_to=self.vehicle)

        # IMU
        imu_bp = self.bp_lib.find('sensor.other.imu')
        imu_loc = {"x": 0.1, "y": 0.0, "z": 2.1, "roll": 0.0, "pitch": 0.0, "yaw": 0.0}
        imu_bp.set_attribute('noise_accel_stddev_x', '0.0')  # Acceleration noise (m/s²)
        imu_bp.set_attribute('noise_accel_stddev_y', '0.0')
        imu_bp.set_attribute('noise_accel_stddev_z', '0.0')
        imu_bp.set_attribute('noise_gyro_stddev_x', '0.0')  # Gyroscope noise (rad/s)
        imu_bp.set_attribute('noise_gyro_stddev_y', '0.0')
        imu_bp.set_attribute('noise_gyro_stddev_z', '0.0')
        imu_bp.set_attribute('noise_gyro_bias_x', '0.0')  # Gyroscope noise (rad/s)
        imu_bp.set_attribute('noise_gyro_bias_y', '0.0')
        imu_bp.set_attribute('noise_gyro_bias_z', '0.0')
        imu_init_trans = carla.Transform(
                                carla.Location(x=imu_loc["x"], y=imu_loc["y"], z=imu_loc["z"]),
                                carla.Rotation(roll=imu_loc["roll"], pitch=imu_loc["pitch"], yaw=imu_loc["yaw"]))
        self.imu = self.world.spawn_actor(imu_bp, imu_init_trans, attach_to=self.vehicle)

        # Collision
        coll_bp = self.bp_lib.find('sensor.other.collision')
        coll_loc = {"x": 0.0, "y": 0.0, "z": 0.0}
        coll_init_trans = carla.Transform(
                                carla.Location(x=coll_loc["x"], y=coll_loc["y"], z=coll_loc["z"]))
        self.collision = self.world.spawn_actor(coll_bp, coll_init_trans, attach_to=self.vehicle)

        # Collision
        lane_bp = self.bp_lib.find('sensor.other.lane_invasion')
        lane_loc = {"x": 0.0, "y": 0.0, "z": 0.0}
        lane_init_trans = carla.Transform(
                                carla.Location(x=lane_loc["x"], y=lane_loc["y"], z=lane_loc["z"]))
        self.lane_invasion = self.world.spawn_actor(lane_bp, lane_init_trans, attach_to=self.vehicle)

        # ODOM
        # odom_bp = self.bp_lib.find('sensor.other.odometry')
        # odom_loc = {"x": 0.0, "y": 0.0, "z": 0.0, "roll": 0.0, "pitch": 0.0, "yaw": 0.0}
        # odom_init_trans = carla.Transform(
        #                         carla.Location(x=odom_loc["x"], y=odom_loc["y"], z=odom_loc["z"]),
        #                         carla.Rotation(roll=odom_loc["roll"], pitch=odom_loc["pitch"], yaw=odom_loc["yaw"]))
        # self.odom = self.world.spawn_actor(odom_bp, odom_init_trans, attach_to=self.vehicle)

        # RADAR
        radar_bp = self.bp_lib.find('sensor.other.radar') 
        radar_bp.set_attribute('horizontal_fov', '30.0')
        radar_bp.set_attribute('vertical_fov', '30.0')
        radar_bp.set_attribute('points_per_second', '10000')
        radar_init_trans = carla.Transform(carla.Location(z=2))
        self.radar = self.world.spawn_actor(radar_bp, radar_init_trans, attach_to=self.vehicle)

        self.publish_base_link_tf()

    def sensor_listen(self):
        self.lidar.listen(lambda data: self.lidar_callback(data, self.point_list))
        self.radar.listen(lambda data: self.radar_callback(data, self.radar_list))
        self.camera.listen(lambda image: self.camera_callback(image, self.camera_data))
        self.gnss.listen(lambda data: self.gnss_callback(data, self.gnss_data))
        self.imu.listen(lambda data: self.imu_callback(data, self.imu_data))

    def sensor_visualization(self):
        cv2.namedWindow('RGB Camera', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RGB Camera', self.camera_data['image'])
        cv2.waitKey(1)

        # Update geometry and camera in game loop
        frame = 0
        while True:
            if frame == 2:
                self.vis.add_geometry(self.point_list)
                self.vis.add_geometry(self.radar_list)
            self.vis.update_geometry(self.point_list)
            self.vis.update_geometry(self.radar_list)
            
            self.vis.poll_events()
            self.vis.update_renderer()
            # # This can fix Open3D jittering issues:
            time.sleep(0.005)
            frame += 1

            cv2.imshow('RGB Camera', self.camera_data['image'])
            # self.vehicle_control()

            # Break if user presses 'q'
            if cv2.waitKey(1) == ord('q'):
                break

    def vehicle_destroy(self):
        # Close displays and stop sensors
        cv2.destroyAllWindows()
        self.radar.stop()
        self.radar.destroy()
        self.lidar.stop()
        self.lidar.destroy()
        self.camera.stop()
        self.camera.destroy()
        self.vis.destroy_window()

        for actor in self.world.get_actors().filter('*vehicle*'):
            actor.destroy()
        for actor in self.world.get_actors().filter('*sensor*'):
            actor.destroy()

    def vehicle_control(self):
        # self._control.throttle
        self.vehicle.apply_control(carla.VehicleControl(throttle=0.5, steer=0.0, brake=0.0))

    def get_ros_image(self, carla_camera_data):
        """
        Function to transform the received carla camera data into a ROS image message
        """
        if ((carla_camera_data.height != self.camera_info.height) or
                (carla_camera_data.width != self.camera_info.width)):
            self.node.logerr(
                "Camera{} received image not matching configuration".format(self.get_prefix()))
        image_data_array, encoding = self.get_carla_image_data_array(
            carla_camera_data)
        img_msg = self.br.cv2_to_imgmsg(image_data_array, encoding=encoding)
        # the camera data is in respect to the camera's own frame
        img_msg.header = self.get_msg_header(timestamp=carla_camera_data.timestamp)

        return img_msg

    @abstractmethod
    def get_carla_image_data_array(self, carla_image):
        carla_image_data_array = np.ndarray(
            shape=(carla_image.height, carla_image.width, 4),
            dtype=np.uint8, buffer=carla_image.raw_data)

        return carla_image_data_array, 'bgra8'

def main(args=None):
    rclpy.init(args=args)
    node = LiosamCarla()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()