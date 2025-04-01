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

from sensor_msgs.msg import CameraInfo, Image, PointCloud2, PointField
from std_msgs.msg import Header

import struct
import ctypes
import ros_compatibility as roscomp
from abc import abstractmethod

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

'''
# LIDAR and RADAR callbacks
def lidar_callback(point_cloud, point_list):
    """Prepares a point cloud with intensity colors ready to be consumed by Open3D"""
    print(point_cloud)
    print("**************** \n")
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

def radar_callback(data, point_list):
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
'''

class EgoVehicle(Node):
    def __init__(self):
        super().__init__('carla_ego_vehicle')

        #CARLA INIT
        self.client = carla.Client('localhost', 2000) 
        self.world = self.client.get_world()
        self.bp_lib = self.world.get_blueprint_library() 
        self.br = CvBridge()
        self.vehicle = None
        self.lidar = None
        self.radar = None
        self.camera = None
        self.camera_data = None
        self.carla_actor = None 
        self.camera_info = None

        self.img_pub = self.create_publisher(Image, 'carla/img', 10)
        self.lidar_pub = self.create_publisher(PointCloud2, 'carla/lidar', 10)
        self.radar_pub = self.create_publisher(PointCloud2, 'carla/radar', 10)
        self.info_pub = self.create_publisher(CameraInfo, 'carla/camera_info', 10)

        # Add auxilliary data structures
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
        point_cloud_msg = create_cloud(header, fields, lidar_data)
        self.lidar_pub.publish(point_cloud_msg)

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
            v.set_autopilot(False) 
        

        # Set up LIDAR and RADAR, parameters are to assisst visualisation
        lidar_bp = self.bp_lib.find('sensor.lidar.ray_cast') 
        lidar_bp.set_attribute('range', '100.0')
        lidar_bp.set_attribute('noise_stddev', '0.1')
        lidar_bp.set_attribute('upper_fov', '15.0')
        lidar_bp.set_attribute('lower_fov', '-25.0')
        lidar_bp.set_attribute('channels', '64.0')
        lidar_bp.set_attribute('rotation_frequency', '20.0')
        lidar_bp.set_attribute('points_per_second', '500000')
            
        lidar_init_trans = carla.Transform(carla.Location(z=2))
        self.lidar = self.world.spawn_actor(lidar_bp, lidar_init_trans, attach_to=self.vehicle)

        radar_bp = self.bp_lib.find('sensor.other.radar') 
        radar_bp.set_attribute('horizontal_fov', '30.0')
        radar_bp.set_attribute('vertical_fov', '30.0')
        radar_bp.set_attribute('points_per_second', '10000')
        radar_init_trans = carla.Transform(carla.Location(z=2))
        self.radar = self.world.spawn_actor(radar_bp, radar_init_trans, attach_to=self.vehicle)

        # Spawn camera
        camera_bp = self.bp_lib.find('sensor.camera.rgb') 
        camera_init_trans = carla.Transform(carla.Location(z=1.5,x=1.1), carla.Rotation())
        self.camera = self.world.spawn_actor(camera_bp, camera_init_trans, attach_to=self.vehicle)
        image_w = camera_bp.get_attribute("image_size_x").as_int()
        image_h = camera_bp.get_attribute("image_size_y").as_int()
        fov = camera_bp.get_attribute("fov")
        self._build_camera_info(image_w, image_h, fov)
        self.camera_data = {'image': np.zeros((image_h, image_w, 4))} 

    def sensor_listen(self):
        self.lidar.listen(lambda data: self.lidar_callback(data, self.point_list))
        self.radar.listen(lambda data: self.radar_callback(data, self.radar_list))
        self.camera.listen(lambda image: self.camera_callback(image, self.camera_data))

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
            self.vehicle_control()

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
    node = EgoVehicle()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()