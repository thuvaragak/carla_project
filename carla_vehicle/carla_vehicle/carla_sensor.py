import sys
import rclpy
from rclpy.node import Node
import ros_compatibility as roscomp
from sensor_msgs.msg import PointCloud2, PointField, Imu
from std_msgs.msg import Header 
from geometry_msgs.msg import TransformStamped, Vector3, Twist, PoseStamped, Pose
from nav_msgs.msg import Path
from obstacle_avoidance.msg import Control

import carla
import random
import numpy as np
import open3d as o3d
import struct

import ctypes
import matplotlib.cm as cm

from tf2_ros import StaticTransformBroadcaster
import threading
from typing import List

import sys

sys.path.extend([
    '/home/thuvaraga/carla_project/CARLA_0.9.14/PythonAPI',
    '/home/thuvaraga/carla_project/CARLA_0.9.14/PythonAPI/carla',
    '/home/thuvaraga/carla_project/CARLA_0.9.14/PythonAPI/carla/agents'
])
from agents.navigation.global_route_planner import GlobalRoutePlanner

VIRIDIS = np.array(cm.get_cmap('plasma').colors)
VID_RANGE = np.linspace(0.0, 1.0, VIRIDIS.shape[0])
COOL_RANGE = np.linspace(0.0, 1.0, VIRIDIS.shape[0])
COOL = np.array(cm.get_cmap('winter')(COOL_RANGE))
COOL = COOL[:, :3]

_DATATYPES = {}
_DATATYPES[PointField.INT8] = ('b', 1)
_DATATYPES[PointField.UINT8] = ('B', 1)
_DATATYPES[PointField.INT16] = ('h', 2)
_DATATYPES[PointField.UINT16] = ('H', 2)
_DATATYPES[PointField.INT32] = ('i', 4)
_DATATYPES[PointField.UINT32] = ('I', 4)
_DATATYPES[PointField.FLOAT32] = ('f', 4)
_DATATYPES[PointField.FLOAT64] = ('d', 8)

V_MAX = 40.0   # must match cfg_.v_max
W_MAX = 1.5    # must match cfg_.w_max

def _get_struct_fmt(is_bigendian, fields, field_names=None):
    fmt = '>' if is_bigendian else '<'

    offset = 0
    for field in (f for f in sorted(fields, key=lambda f: f.offset)
                  if field_names is None or f.name in field_names):
        if offset < field.offset:
            fmt += 'x' * (field.offset - offset)
            offset = field.offset
        if field.datatype not in _DATATYPES:
            print('Skipping unknown PointField datatype [{}]'.format(field.datatype))
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

class PathPlanner:
    def __init__(self, world, start: carla.Location, end: carla.Location):
        self.world = world
        self.map = world.get_map()
        self.gr_planner = GlobalRoutePlanner(self.map, 1.0)
        self.waypoints = self._compute_route(start, end)
        self.cur_wp_index = 0
        
    def _compute_route(self, start: carla.Location, end: carla.Location) -> List[carla.Waypoint]:
        route = self.gr_planner.trace_route(start, end)
        return [waypoint for waypoint, _ in route]
    
    def publish_path(self) -> Path:
        path_msg = Path()
        path_msg.header.stamp = rclpy.time.Time().to_msg()
        path_msg.header.frame_id = 'map'
        
        for wp in self.waypoints:
            pose = PoseStamped()
            pose.header.stamp = rclpy.time.Time().to_msg()
            pose.header.frame_id = 'map'
            pose.pose.position.x = wp.transform.location.x
            pose.pose.position.y = wp.transform.location.y
            pose.pose.position.z = wp.transform.location.z
            
            # quat = wp.transform.rotation
            # pose.pose.orientation.x = quat.x
            # pose.pose.orientation.y = quat.y
            # pose.pose.orientation.z = quat.z
            # pose.pose.orientation.w = quat.w
            
            path_msg.poses.append(pose)
        
        return path_msg
    
class CarlaSensor(Node):
    def __init__(self):
        super().__init__('carla_sensor')
        try:
            self.client = carla.Client('localhost', 2000)
            self.client.set_timeout(10.0)  # 10 second timeout
            self.world = self.client.get_world()
            self.bp_lib = self.world.get_blueprint_library()
        except Exception as e:
            self.get_logger().error(f'Failed to connect to CARLA: {e}')
            raise

        self.path_planner = PathPlanner(
            self.world, 
            carla.Location(x=0.0, y=16.905891, z=0.600000),
            carla.Location(x=500, y=20, z=0)
        )
        # self.path_planner.publish_path()
        
        # Publishers
        self.lidar_pub = self.create_publisher(PointCloud2, 'lidar/data', 10)
        self.imu_pub = self.create_publisher(Imu, '/imu/data', 10)
        self.vel_pub = self.create_publisher(Vector3, '/vel/data', 10)
        self.path_pub = self.create_publisher(Path, '/global_plan', 10)
        self.pos_pub = self.create_publisher(Pose, '/vehicle/pose', 10)
        
        # Subscriber
        self.subscription = self.create_subscription(
            Control, '/control', self.control_loop, 10)
        
        # TF Broadcasters
        self.tf_chassis = StaticTransformBroadcaster(self)
        self.tf_lidar = StaticTransformBroadcaster(self)
        self.tf_imu = StaticTransformBroadcaster(self)
        
        # State variables
        self.vehicle = None
        self.lidar = None
        self.imu = None
        self.camera = None
        self.image = None
        self.point_list = o3d.geometry.PointCloud()
        self.spectator = None
        self._destroyed = False
        self._control_lock = threading.Lock()
        
        # Parameters
        self.declare_parameter('lidar_range', 100.0)
        self.declare_parameter('lidar_frequency', 20.0)
        self.declare_parameter('imu_frequency', 100.0)
        self.declare_parameter('spawn_npcs', True)
        
        try:
            self.vehicle_setup()
            self.sensor_listen()
            self.get_logger().info('CARLA sensor node initialized successfully')
        except Exception as e:
            self.get_logger().error(f'Failed to initialize: {e}')
            self.vehicle_destroy()
            raise

    def vehicle_setup(self):
        """Setup vehicle and sensors"""
        spawn_points = self.world.get_map().get_spawn_points()
        
        # Spawn main vehicle
        vehicle_bp = self.bp_lib.find('vehicle.audi.a2')
        if not vehicle_bp:
            raise Exception("Audi A2 blueprint not found")
            
        vehicle_bp.set_attribute('role_name', 'hero')
        spawn_tf = carla.Transform(
                carla.Location(x=0.0, y=16.905891, z=0.600000),
                carla.Rotation(yaw=-179.840790)
            )
        self.vehicle = self.world.spawn_actor(vehicle_bp, spawn_tf)
        
        if not self.vehicle:
            raise Exception("Failed to spawn vehicle")
        
        # Move spectator to follow vehicle
        self.spectator = self.world.get_spectator()

        # Spawn NPC vehicles if enabled
        if self.get_parameter('spawn_npcs').value:
            self._spawn_npcs(spawn_points)

        # Setup sensors
        self._setup_lidar()
        self._setup_imu()
        
        # Broadcast TFs
        self.broadcasting_tf()

    def _spawn_npcs(self, spawn_points):
        """Spawn NPC vehicles"""
        for i in range(100):  # Reduced from 100 for performance
            vehicle_bp = random.choice(self.bp_lib.filter('vehicle'))
            npc = self.world.try_spawn_actor(vehicle_bp, random.choice(spawn_points))
            if npc:
                npc.set_autopilot(True)

    def _setup_lidar(self):
        """Configure and spawn LiDAR sensor"""
        lidar_bp = self.bp_lib.find('sensor.lidar.ray_cast')
        if not lidar_bp:
            raise Exception("LiDAR blueprint not found")
            
        # Set LiDAR parameters
        lidar_range = self.get_parameter('lidar_range').value
        lidar_frequency = self.get_parameter('lidar_frequency').value
        
        lidar_bp.set_attribute('range', str(lidar_range))
        lidar_bp.set_attribute('noise_stddev', '0.0')
        lidar_bp.set_attribute('upper_fov', '3.0')
        lidar_bp.set_attribute('lower_fov', '-27.0')
        lidar_bp.set_attribute('channels', '64.0')
        lidar_bp.set_attribute('rotation_frequency', str(lidar_frequency))
        lidar_bp.set_attribute('points_per_second', '480000')
        
        loc = {"x": 0.0, "y": 0.0, "z": 2.100, "roll": 0.0, "pitch": 0.0, "yaw": 0.0}
        lidar_init_trans = carla.Transform(
            carla.Location(x=loc["x"], y=loc["y"], z=loc["z"]),
            carla.Rotation(roll=loc["roll"], pitch=loc["pitch"], yaw=loc["yaw"])
        )
        self.lidar = self.world.spawn_actor(lidar_bp, lidar_init_trans, attach_to=self.vehicle)

    def _setup_camera(self):
        camera_bp = self.bp_lib.find('sensor.camera.semantic_segmentation')
        if not camera_bp:
            raise Exception("Semantic Segmentation Camera blueprint not found")
            
        # Set Semantic Segmentation Camera parameters
        camera_bp.set_attribute("image_size_x", "640")
        camera_bp.set_attribute("image_size_y", "480")
        camera_bp.set_attribute("fov", "90")

        loc = {"x": 0.0, "y": 0.0, "z": 2.100, "roll": 0.0, "pitch": 0.0, "yaw": 0.0}
        camera_init_trans = carla.Transform(carla.Location(x=1.80, y=0.25, z=1.65))
        self.camera = self.world.spawn_actor(camera_bp, camera_init_trans, attach_to=self.vehicle)

    def _setup_imu(self):
        """Configure and spawn IMU sensor"""
        imu_bp = self.bp_lib.find('sensor.other.imu')
        if not imu_bp:
            raise Exception("IMU blueprint not found")
            
        imu_frequency = self.get_parameter('imu_frequency').value
        imu_bp.set_attribute('sensor_tick', str(1.0/imu_frequency))
        
        # Set noise attributes to zero for clean data
        noise_attributes = [
            'noise_accel_stddev_x', 'noise_accel_stddev_y', 'noise_accel_stddev_z',
            'noise_gyro_stddev_x', 'noise_gyro_stddev_y', 'noise_gyro_stddev_z',
            'noise_gyro_bias_x', 'noise_gyro_bias_y', 'noise_gyro_bias_z'
        ]
        for attr in noise_attributes:
            imu_bp.set_attribute(attr, '0.0')
            
        imu_loc = {"x": 0.1, "y": 0.0, "z": 2.1, "roll": 0.0, "pitch": 0.0, "yaw": 0.0}
        imu_init_trans = carla.Transform(
            carla.Location(x=imu_loc["x"], y=imu_loc["y"], z=imu_loc["z"]),
            carla.Rotation(roll=imu_loc["roll"], pitch=imu_loc["pitch"], yaw=imu_loc["yaw"])
        )
        self.imu = self.world.spawn_actor(imu_bp, imu_init_trans, attach_to=self.vehicle)

    def broadcasting_tf(self):
        """Broadcast static transforms for sensors"""
        # LiDAR TF
        lidar_tf = TransformStamped()
        lidar_tf.header.stamp = self.get_clock().now().to_msg()
        lidar_tf.header.frame_id = 'base_link'
        lidar_tf.child_frame_id = 'lidar_link'
        lidar_tf.transform.translation.x = 1.5
        lidar_tf.transform.translation.y = 0.0
        lidar_tf.transform.translation.z = 2.1
        lidar_tf.transform.rotation.w = 1.0
        self.tf_lidar.sendTransform(lidar_tf)

        # IMU TF
        imu_tf = TransformStamped()
        imu_tf.header.stamp = self.get_clock().now().to_msg()
        imu_tf.header.frame_id = 'base_link'
        imu_tf.child_frame_id = 'imu_link'
        imu_tf.transform.translation.x = 0.1
        imu_tf.transform.translation.y = 0.0
        imu_tf.transform.translation.z = 2.1
        imu_tf.transform.rotation.w = 1.0
        self.tf_imu.sendTransform(imu_tf)

    def get_msg_header(self, timestamp=None):
        header = Header()
        if not timestamp:
            header.stamp = self.get_clock().now().to_msg()
        else:
            header.stamp = roscomp.ros_timestamp(sec=timestamp, from_sec=True)
        return header

    def get_vel_pose(self):
        """Publish vehicle velocity and update spectator"""
        if self._destroyed:
            return
            
        try:
            # Update spectator to follow vehicle
            vehicle_tf = self.vehicle.get_transform()
            spectator_tf = carla.Transform(
                vehicle_tf.location + carla.Location(z=3.0),
                vehicle_tf.rotation
            )
            self.spectator.set_transform(spectator_tf)
            
            # Publish velocity
            velocity = self.vehicle.get_velocity()
            vel = Vector3()
            vel.x = velocity.x
            vel.y = velocity.y
            vel.z = velocity.z
            self.vel_pub.publish(vel)
            self.path_pub.publish(self.path_planner.publish_path())

            # Publish vehicle pose
            pose_msg = Pose()
            pose_msg.position.x = vehicle_tf.location.x
            pose_msg.position.y = vehicle_tf.location.y
            pose_msg.position.z = vehicle_tf.location.z
            pose_msg.orientation.x = vehicle_tf.rotation.roll
            pose_msg.orientation.y = vehicle_tf.rotation.pitch
            pose_msg.orientation.z = vehicle_tf.rotation.yaw
            self.pos_pub.publish(pose_msg)
        
        except Exception as e:
            self.get_logger().warn(f'Error in get_vel_pose: {e}')

    def sensor_listen(self):
        """ Start listening to sensor data """
        self.lidar.listen(self.lidar_callback)
        self.imu.listen(self.imu_callback)
        self.create_timer(0.02, self.get_vel_pose)  # 50 Hz velocity publishing

    def camera_callback(self, image):
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        self.image = array.reshape((480, 640, 4))[:, :, :3]
        
    def lidar_callback(self, point_cloud):
        if self._destroyed:
            return
            
        try:
            # Convert to numpy array
            data = np.frombuffer(point_cloud.raw_data, dtype=np.float32)
            data = np.reshape(data, (int(data.shape[0] / 4), 4)).copy()

            # Create PointCloud2 message
            header = self.get_msg_header(timestamp=point_cloud.timestamp)
            header.frame_id = "lidar_link"

            fields = [
                PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
                PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
                PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
                PointField(name='intensity', offset=12, datatype=PointField.FLOAT32, count=1),
                PointField(name='ring', offset=16, datatype=PointField.UINT16, count=1),
            ]

            # CARLA (left-handed) → ROS (right-handed): flip Y
            lidar_data = data.copy()
            lidar_data[:, 1] *= -1

            num_points = lidar_data.shape[0]
            num_channels = int(float(self.lidar.attributes['channels']))

            # Synthetic ring assignment
            rings = (np.arange(num_points, dtype=np.uint16) % num_channels)

            # Build points for message
            points_for_msg = [
                (float(row[0]), float(row[1]), float(row[2]), float(row[3]), int(rings[i]))
                for i, row in enumerate(lidar_data)
            ]

            point_cloud_msg = create_cloud(header, fields, points_for_msg)
            self.lidar_pub.publish(point_cloud_msg)
            
        except Exception as e:
            self.get_logger().error(f'Error in lidar_callback: {e}')

    def imu_callback(self, data):
        if self._destroyed:
            return
            
        try:
            imu_msg = Imu()
            imu_msg.header.stamp = self.get_clock().now().to_msg()
            imu_msg.header.frame_id = 'imu_link'
            
            # Convert from CARLA to ROS coordinate system
            # CARLA: X=forward, Y=right, Z=up
            # ROS: X=forward, Y=left, Z=up
            imu_msg.linear_acceleration.x = data.accelerometer.x
            imu_msg.linear_acceleration.y = -data.accelerometer.y  # Flip Y
            imu_msg.linear_acceleration.z = data.accelerometer.z
            
            imu_msg.angular_velocity.x = data.gyroscope.x
            imu_msg.angular_velocity.y = -data.gyroscope.y  # Flip Y
            imu_msg.angular_velocity.z = data.gyroscope.z
            
            self.imu_pub.publish(imu_msg)
        except Exception as e:
            self.get_logger().error(f'Error in imu_callback: {e}')

    def control_loop(self, msg):
        print("Control", msg)
        control = carla.VehicleControl()
        control.steer = msg.steer
        control.throttle = msg.throttle
        control.brake = msg.brake
        control.hand_brake = False
        control.reverse = False
        self.vehicle.apply_control(control)
    
    def vehicle_destroy(self):
        """Cleanup CARLA actors"""
        if self._destroyed:
            return
            
        self._destroyed = True
        self.get_logger().info('Destroying CARLA actors...')
        
        try:
            if self.lidar:
                self.lidar.stop()
                self.lidar.destroy()
            if self.imu:
                self.imu.stop()
                self.imu.destroy()
            if self.vehicle:
                self.vehicle.destroy()
                
            # Cleanup other actors
            for actor in self.world.get_actors().filter('*vehicle*'):
                if actor.is_alive:
                    actor.destroy()
            for actor in self.world.get_actors().filter('*sensor*'):
                if actor.is_alive:
                    actor.destroy()
                    
        except Exception as e:
            self.get_logger().error(f'Error during cleanup: {e}')

    def __del__(self):
        """Destructor to ensure cleanup"""
        self.vehicle_destroy()

def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = CarlaSensor()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f'Error: {e}')
    finally:
        if node:
            node.vehicle_destroy()
            node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
