import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import Image
import cv2
from ultralytics import YOLO
from cv_bridge import CvBridge


class ObjDetection(Node):

    def __init__(self):
        super().__init__('obj_detection')

        self.br = CvBridge()
        self.model = YOLO('yolov8n.pt')
        self.subscription = self.create_subscription(
            Image,
            'carla/img',
            self.img_callback,
            10)

    def img_callback(self, msg):
        frame = self.br.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        results = self.model(frame)[0]
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls = int(box.cls[0])
            label = self.model.names[cls]
            conf = float(box.conf[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'{label} {conf:.2f}', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        cv2.imshow("YOLO Detection", frame)
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    obj = ObjDetection()
    rclpy.spin(obj)
    obj.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()