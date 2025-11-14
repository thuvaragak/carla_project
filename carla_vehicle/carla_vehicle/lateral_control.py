# from email.mime import image
import rclpy
from rclpy.node import Node
import carla 
import math 
import random 
import numpy as np
import cv2
from typing import Optional, Tuple, List
import sys
from ultralytics import YOLO
from cv_bridge import CvBridge

sys.path.extend([
    '/home/thuvaraga/carla_project/CARLA_0.9.14/PythonAPI',
    '/home/thuvaraga/carla_project/CARLA_0.9.14/PythonAPI/carla',
    '/home/thuvaraga/carla_project/CARLA_0.9.14/PythonAPI/carla/agents'
])
from agents.navigation.global_route_planner import GlobalRoutePlanner

class DetectedObject:
    """Enhanced object detection result"""
    def __init__(self, bbox: Tuple[int, int, int, int], class_id: int, 
                 confidence: float, class_name: str, distance: float = float('inf')):
        self.bbox = bbox  # (x1, y1, x2, y2)
        self.class_id = class_id
        self.confidence = confidence
        self.class_name = class_name
        self.distance = distance
        self.center_x = (bbox[0] + bbox[2]) // 2
        self.center_y = (bbox[1] + bbox[3]) // 2

class YOLOObstacleDetector:
    """Dedicated YOLO obstacle detection with depth integration"""
    
    def __init__(self, model_path: str = 'yolov8n.pt', confidence_threshold: float = 0.5):
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold
        self.class_names = self.model.names
        
        # Define relevant obstacle classes for autonomous driving
        # , 'traffic light', 'stop sign', 'person', 
        self.obstacle_classes = {
            'bicycle', 'car', 'motorcycle', 'bus', 'person',
            'truck'
        }
    
    def detect_obstacles(self, rgb_image: np.ndarray, depth_map: np.ndarray = None) -> List[DetectedObject]:
        """Detect obstacles and calculate distances using depth map"""
        obstacles = []
        
        try:
            # Run YOLO inference
            results = self.model(rgb_image, verbose=False, conf=self.confidence_threshold)
            
            for result in results:
                boxes = result.boxes
                if boxes is None:
                    continue
                    
                for box in boxes:
                    confidence = box.conf.item()
                    class_id = int(box.cls.item())
                    class_name = self.class_names[class_id]
                    
                    # Filter for relevant obstacle classes
                    if class_name not in self.obstacle_classes:
                        continue
                    
                    # Get bounding box coordinates
                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                    
                    # Calculate distance if depth map is available
                    distance = self._calculate_object_distance(depth_map, x1, y1, x2, y2)
                    
                    obstacle = DetectedObject(
                        bbox=(x1, y1, x2, y2),
                        class_id=class_id,
                        confidence=confidence,
                        class_name=class_name,
                        distance=distance
                    )

                    obstacles.append(obstacle)
                    
        except Exception as e:
            print(f"YOLO detection error: {e}")
            
        return obstacles
    
    def _calculate_object_distance(self, depth_map: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> float:
        """Calculate object distance using depth map"""
        if depth_map is None or depth_map.size == 0:
            return float('inf')
            
        H, W = depth_map.shape
        
        # Ensure coordinates are within bounds
        x1, x2 = max(0, x1), min(W-1, x2)
        y1, y2 = max(0, y1), min(H-1, y2)
        
        if x1 >= x2 or y1 >= y2:
            return float('inf')
        
        # Focus on lower part of bounding box (where object touches ground)
        ground_region_height = int((y2 - y1) * 0.3)  # Bottom 30%
        ground_y1 = y2 - ground_region_height
        ground_y1 = max(y1, ground_y1)
        
        if ground_y1 >= y2:
            return float('inf')
            
        # Extract depth values from ground contact region
        ground_region = depth_map[y1:y2, x1:x2]
        
        if ground_region.size == 0:
            return float('inf')
            
        # Use median to avoid outliers
        valid_depths = ground_region[ground_region > 0]
        if valid_depths.size == 0:
            return float('inf')
            
        return float(np.median(valid_depths))

class OptimizedPID:
    """Optimized PID controller with derivative filtering"""
    def __init__(self, kp: float = 1.0, ki: float = 0.5, kd: float = 0.5, 
                 i_lim: float = 0.3, dt: float = 0.05, alpha: float = 0.3):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.i_lim = i_lim
        self.dt = dt
        self.alpha = alpha
        
        self.integral = 0.0
        self.prev_error = 0.0
        self.prev_derivative = 0.0
        
    def reset(self):
        self.integral = 0.0
        self.prev_error = 0.0
        self.prev_derivative = 0.0
        
    def step(self, error: float) -> float:
        # Anti-windup integral
        self.integral = np.clip(self.integral + error * self.dt, -self.i_lim, self.i_lim)
        
        # Filtered derivative
        raw_derivative = (error - self.prev_error) / max(self.dt, 1e-6)
        derivative = self.alpha * raw_derivative + (1 - self.alpha) * self.prev_derivative
        
        self.prev_error = error
        self.prev_derivative = derivative
        
        return self.kp * error + self.ki * self.integral + self.kd * derivative

class DepthProcessor:
    """Enhanced depth processing with YOLO integration"""
    def __init__(self, width: int = 960, height: int = 540):
        self.W = width
        self.H = height
        self.fx = width / (2.0 * math.tan(math.radians(45)))
        self.fy = self.fx
        self.cx = width * 0.5
        self.cy = height * 0.5
        self.br = CvBridge()
        
    def convert_to_meters(self, carla_image) -> np.ndarray:
        """Convert CARLA depth image to meters efficiently"""
        try:
            carla_image.convert(carla.ColorConverter.LogarithmicDepth)
            array = np.frombuffer(carla_image.raw_data, dtype=np.uint8)
            array = array.reshape((self.H, self.W, 4))
            
            # Vectorized conversion to meters
            rgb = array[:, :, :3].astype(np.float32)
            normalized = (rgb[:, :, 2] + rgb[:, :, 1] * 256.0 + 
                         rgb[:, :, 0] * 65536.0) / (256.0**3 - 1.0)
            return 1000.0 * normalized
            
        except Exception as e:
            print(f"Depth conversion error: {e}")
            return np.zeros((self.H, self.W), dtype=np.float32)

class PathFollower:
    def __init__(self, world, start: carla.Location, end: carla.Location):
        self.world = world
        self.map = world.get_map()
        self.gr_planner = GlobalRoutePlanner(self.map, 1.0)
        self.waypoints = self._compute_route(start, end)
        self.cur_wp_index = 0
        
    def _compute_route(self, start: carla.Location, end: carla.Location) -> List[carla.Waypoint]:
        route = self.gr_planner.trace_route(start, end)
        return [waypoint for waypoint, _ in route]
    
    def get_target_point(self, vehicle_tf: carla.Transform, lookahead: float = 10.0) -> carla.Location:
        """Get target point using efficient carrot following"""
        vehicle_loc = vehicle_tf.location
        
        # Find nearest point on path
        nearest_idx, projection, min_dist, param = self._find_nearest_point(vehicle_loc.x, vehicle_loc.y)
        
        if nearest_idx is None:
            return self.waypoints[-1].transform.location
            
        # Move along path
        current_idx = nearest_idx
        remaining = lookahead + min_dist
        
        while current_idx < len(self.waypoints) - 1 and remaining > 0:
            wp1 = self.waypoints[current_idx].transform.location
            wp2 = self.waypoints[current_idx + 1].transform.location
            
            segment_vec = carla.Vector3D(wp2.x - wp1.x, wp2.y - wp1.y, wp2.z - wp1.z)
            segment_length = math.sqrt(segment_vec.x**2 + segment_vec.y**2)
            
            if remaining <= segment_length:
                ratio = remaining / segment_length
                wp =carla.Location(
                    x=wp1.x + ratio * segment_vec.x,
                    y=wp1.y + ratio * segment_vec.y,
                    z=wp1.z + ratio * segment_vec.z
                )
                # print(current_idx)
                # print("Target Point:", wp.x, wp.y, wp.z)
                # print(len(self.waypoints))
                # print("----")
                return wp
                
            remaining -= segment_length
            current_idx += 1
            
        return self.waypoints[-1].transform.location
    
    def _find_nearest_point(self, px: float, py: float) -> Tuple[Optional[int], Tuple[float, float], float, float]:
        """Find nearest point on path efficiently"""
        min_dist_sq = float('inf')
        best_idx, best_point, best_param = None, (0.0, 0.0), 0.0
        
        for i in range(len(self.waypoints) - 1):
            wp1 = self.waypoints[i].transform.location
            wp2 = self.waypoints[i + 1].transform.location
            
            seg_vec = (wp2.x - wp1.x, wp2.y - wp1.y)
            pt_vec = (px - wp1.x, py - wp1.y)
            seg_len_sq = seg_vec[0]**2 + seg_vec[1]**2
            
            if seg_len_sq < 1e-6:
                proj, param = (wp1.x, wp1.y), 0.0
            else:
                param = max(0.0, min(1.0, (pt_vec[0]*seg_vec[0] + pt_vec[1]*seg_vec[1]) / seg_len_sq))
                proj = (wp1.x + param * seg_vec[0], wp1.y + param * seg_vec[1])
                
            dist_sq = (px - proj[0])**2 + (py - proj[1])**2
            
            if dist_sq < min_dist_sq:
                min_dist_sq, best_idx, best_point, best_param = dist_sq, i, proj, param
                
        return best_idx, best_point, math.sqrt(min_dist_sq), best_param

class EnhancedCarlaPID(Node):
    def __init__(self):
        super().__init__('enhanced_carla_pid')
        
        # CARLA connection
        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(10.0)
        self.world = self.client.get_world()
        self.blueprint_lib = self.world.get_blueprint_library()
        
        # Enable synchronous mode
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05
        self.world.apply_settings(settings)
        
        # Core components
        self.yolo_detector = YOLOObstacleDetector()
        self.depth_processor = DepthProcessor()
        self.pid_controller = OptimizedPID()
        self.path_follower = PathFollower(
            self.world, 
            carla.Location(x=0.0, y=16.905891, z=0.600000),
            carla.Location(x=500, y=20, z=0)
        )
        
        # State variables
        self.depth_map = np.zeros((540, 960), dtype=np.float32)
        self.rgb_image = np.zeros((540, 960, 3), dtype=np.uint8)
        self.seg_image = np.zeros((540, 960, 4), dtype=np.uint8)
        self.detected_obstacles = []
        self.safe_distance_base = 15.0
        
        # Camera parameters
        self.fx = 960 / (2.0 * math.tan(math.radians(45)))
        self.cx = 960 * 0.5
        
        # Initialize vehicle and sensors
        self._initialize_vehicle_and_sensors()
        
        # Control timer
        self.control_timer = self.create_timer(0.05, self._control_cycle)
        
        self.get_logger().info("Enhanced CARLA PID with YOLO Ready!")

    def _initialize_vehicle_and_sensors(self):
        """Initialize vehicle, RGB camera, depth camera, and semantic segmentation camera"""
        try:
            # Spawn minimal traffic
            self._spawn_traffic(50)

            # Spawn vehicle
            vehicle_bp = self.blueprint_lib.find('vehicle.audi.a2')
            spawn_tf = carla.Transform(
                carla.Location(x=0.0, y=16.905891, z=0.600000),
                carla.Rotation(yaw=-179.840790)
            )
            self.vehicle = self.world.spawn_actor(vehicle_bp, spawn_tf)
            self.spectator = self.world.get_spectator()
            
            # Setup RGB camera
            rgb_bp = self.blueprint_lib.find("sensor.camera.rgb")
            rgb_bp.set_attribute("image_size_x", "960")
            rgb_bp.set_attribute("image_size_y", "540")
            rgb_bp.set_attribute("fov", "90")
            
            # Setup depth camera
            depth_bp = self.blueprint_lib.find("sensor.camera.depth")
            depth_bp.set_attribute("image_size_x", "960")
            depth_bp.set_attribute("image_size_y", "540")
            depth_bp.set_attribute("fov", "90")
            
            camera_tf = carla.Transform(carla.Location(x=1.80, y=0.25, z=1.65))
            
            self.rgb_camera = self.world.spawn_actor(
                rgb_bp, camera_tf, attach_to=self.vehicle)
            self.depth_camera = self.world.spawn_actor(
                depth_bp, camera_tf, attach_to=self.vehicle)

            self.rgb_camera.listen(self._rgb_callback)
            self.depth_camera.listen(self._depth_callback)
            
        except Exception as e:
            self.get_logger().error(f"Initialization failed: {e}")
            raise

    def _spawn_traffic(self, count: int):
        """Spawn traffic for testing"""
        spawn_points = self.world.get_map().get_spawn_points()
        for i in range(count): 
            vehicle_bp = random.choice(self.blueprint_lib.filter('vehicle')) 
            npc = self.world.try_spawn_actor(vehicle_bp, random.choice(spawn_points))    
        for v in self.world.get_actors().filter('*vehicle*'): 
            v.set_autopilot(False) 

    def _spawn_traffic(self, count, exclude_player_vehicle=True):
        spawn_points = self.world.get_map().get_spawn_points()
        spawned_vehicles = []
        
        # Get existing vehicles if we want to exclude player vehicle
        existing_vehicles = []
        if exclude_player_vehicle:
            existing_vehicles = [v.id for v in self.world.get_actors().filter('*vehicle*')]
        
        for i in range(count):
            vehicle_bp = random.choice(self.blueprint_lib.filter('vehicle'))
            spawn_point = random.choice(spawn_points)
            
            vehicle = self.world.try_spawn_actor(vehicle_bp, spawn_point)
            
            if vehicle is not None:
                spawned_vehicles.append(vehicle)
                vehicle.set_autopilot(True)
            else:
                print(f"Failed to spawn vehicle {i+1}/{count}")
        
        print(f"Successfully spawned {len(spawned_vehicles)} NPC vehicles")
        return spawned_vehicles

    def _rgb_callback(self, image):
        """RGB camera callback"""
        try:
            array = np.frombuffer(image.raw_data, dtype=np.uint8)
            self.rgb_image = array.reshape((540, 960, 4))[:, :, :3]
        except Exception as e:
            self.get_logger().error(f"RGB callback error: {e}")

    def _depth_callback(self, image):
        """Depth camera callback"""
        try:
            self.depth_map = self.depth_processor.convert_to_meters(image)
        except Exception as e:
            self.get_logger().error(f"Depth callback error: {e}")

    def _get_vehicle_speed(self) -> float:
        """Get vehicle speed"""
        velocity = self.vehicle.get_velocity()
        return math.sqrt(velocity.x**2 + velocity.y**2)

    def _compute_steering(self, target_pixel: int) -> float:
        """Compute steering command using PID"""
        if target_pixel is None:
            return 0.0
            
        error = (target_pixel - self.cx) / self.fx
        error = math.atan(error)  # Convert to angle
        
        return float(np.clip(self.pid_controller.step(error), -1.0, 1.0))

    def _compute_throttle_brake(self, obstacles: List[DetectedObject], nav: bool) -> Tuple[float, float]:
        """Compute throttle and brake based on YOLO obstacles"""
        if not obstacles:
            return 0.6, 0.0  # Normal speed
            
        # Find closest obstacle
        closest_distance = min((obs.distance for obs in obstacles), default=float('inf')) / 100

        # Adaptive control based on obstacle distance
        if closest_distance < 1 and nav == True:
            print("Emergency Brake Activated")
            return 0.0, 1.0  # Emergency brake
        elif closest_distance < 3 and nav == True:
            return 0.2, 0.0  
        elif closest_distance < 7 and nav == True:
            return 0.4, 0.0  
        elif closest_distance < 15 and nav == True:       
            print("Reduced Speed Activated")
            return 0.5, 0.0 
        else:
            return 0.6, 0.0  # Normal speed

    def _get_desired_direction(self) -> int:
        vehicle_tf = self.vehicle.get_transform()
        target_loc = self.path_follower.get_target_point(vehicle_tf, 10.0)
        
        # Transform to vehicle coordinates
        yaw = math.radians(vehicle_tf.rotation.yaw)
        dx = target_loc.x - vehicle_tf.location.x
        dy = target_loc.y - vehicle_tf.location.y
        
        x_rel = math.cos(yaw) * dx + math.sin(yaw) * dy
        y_rel = -math.sin(yaw) * dx + math.cos(yaw) * dy
        
        desired_heading = math.atan2(y_rel, x_rel)
        
        # Convert to pixel column
        return int(self.cx + self.fx * math.tan(desired_heading))

    def safe_gap(self, detected_obstacles: List[DetectedObject], target_pixel: int):
        obs_left = []
        obs_right = []
        
        if len(detected_obstacles) > 0:
            for obs in detected_obstacles:
                if obs.distance/100.0 <=20:
                    x1, y1, x2, y2 = obs.bbox
                    a = (x1 + x2)/2
                    dx = a - target_pixel
                    if dx>0:
                        obs_right.append(dx)
                    else:
                        obs_left.append(dx)
        m=0
        n = 0
        nav = False
        avg = 0
        if len(obs_left) >0:
            m = max(obs_left)
        if len(obs_right) >0:
            n = min(obs_right)
        if len(obs_left) >0 and len(obs_right) >0:
            avg = (n-m)/2
            if n < 100 and  abs(m)< 100: 
                nav = True
            elif n<70:
                if abs(m) >70 and abs(m)< 250:
                    nav = True
                    target_pixel = target_pixel - 100
            elif abs(m)<100:
                if n >100 and n < 200:
                    nav = True
                    target_pixel = target_pixel + avg + m
            else:
                nav = False
                if n < 200 or abs(m)< 200: 
                    target_pixel = target_pixel + avg + m

        elif len(obs_left) >0 and len(obs_right) ==0:
            if abs(m)<70:
                nav = True
                target_pixel = target_pixel + 100
        elif len(obs_left) ==0 and len(obs_right) >0:
            if n<70:
                nav = True
                target_pixel = target_pixel - 100
        else:
            nav = False

        return nav, target_pixel

    def _control_cycle(self):
        """Main control cycle with YOLO obstacle detection"""
        try:
            vehicle_tf = self.vehicle.get_transform()
            spectator_tf = carla.Transform(
                vehicle_tf.location + carla.Location(z=3.0),
                vehicle_tf.rotation
            )
            self.spectator.set_transform(spectator_tf)
            self.world.tick()
            
            # 1. YOLO Obstacle Detection
            self.detected_obstacles = self.yolo_detector.detect_obstacles(
                self.rgb_image, self.depth_map)
            
            # 2. Get current state
            speed = self._get_vehicle_speed()
            
            # 3. Pure pursuit 
            target_pixel = self._get_desired_direction()
            
            # 4. Safe Gap
            nav, target_pixel = self.safe_gap(self.detected_obstacles, target_pixel)
                
            # 5. Compute Steering control using PID
            steering = self._compute_steering(target_pixel)

            # 6. Compute Throttle and Brake
            throttle, brake = self._compute_throttle_brake( self.detected_obstacles, nav)

            # 7. Apply control
            control = carla.VehicleControl()
            control.throttle = throttle
            control.steer = steering
            control.brake = brake
            self.vehicle.apply_control(control)
            
            # 8. Visualization
            self._update_visualization(int(target_pixel), self.detected_obstacles)
            
        except Exception as e:
            self.get_logger().error(f"Control cycle error: {e}")

    def _update_visualization(self, target_pixel: Optional[int], 
                            obstacles: List[DetectedObject]):
        """Enhanced visualization with YOLO detections"""
        try:
            display_img = self.rgb_image.copy()
            
            # Draw target guidance
            if target_pixel is not None:
                cv2.line(display_img, (target_pixel, 0), (target_pixel, 540), 
                        (0, 255, 0), 3)
                cv2.putText(display_img, "TARGET", (target_pixel-30, 30),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Draw YOLO detections with distance-based coloring
            for obstacle in obstacles:
                x1, y1, x2, y2 = obstacle.bbox
                
                # Color coding by distance
                if obstacle.distance < 10.0:
                    color = (0, 0, 255)  # Red - very close
                elif obstacle.distance < 20.0:
                    color = (0, 165, 255)  # Orange - close
                else:
                    color = (0, 255, 255)  # Yellow - far
                
                # Draw bounding box
                cv2.rectangle(display_img, (x1, y1), (x2, y2), color, 2)
                
                # Draw info
                info_text = f"{obstacle.class_name} {obstacle.confidence:.2f} {obstacle.distance/100:.1f}m"
                cv2.putText(display_img, info_text, (x1, y1-10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            # Add status info
            status_text = f"Obstacles: {len(obstacles)}"
            cv2.putText(display_img, status_text, (10, 50),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.imshow("CARLA Lateral Control", display_img)
            cv2.waitKey(1)
            
        except Exception as e:
            self.get_logger().error(f"Visualization error: {e}")

    def destroy_node(self):
        """Cleanup"""
        try:
            sensors = [self.rgb_camera, self.depth_camera]
            for sensor in sensors:
                if sensor and sensor.is_alive:
                    sensor.stop()
                    sensor.destroy()
            
            if self.vehicle and self.vehicle.is_alive:
                self.vehicle.destroy()
                
            cv2.destroyAllWindows()
            
        except Exception as e:
            self.get_logger().error(f"Cleanup error: {e}")
        
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    controller = EnhancedCarlaPID()
    
    try:
        rclpy.spin(controller)
    except KeyboardInterrupt:
        pass
    finally:
        controller.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()