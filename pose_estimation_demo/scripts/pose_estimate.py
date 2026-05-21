import numpy as np
import torch
import open3d as o3d

# アルゴリズムの切り替え（将来的に色ありに戻す場合はここを書き換える）
import Registration_SVD as Registration 

from sensor_msgs_py import point_cloud2 
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import TransformStamped
from tf2_msgs.msg import TFMessage
import tf2_ros
from tf2_ros import TransformBroadcaster
import transformations as tf_transformations
import os
import copy
from scipy.spatial.transform import Rotation as R
import time
from std_msgs.msg import Header

def create_pcd(xyz):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    return pcd

class PointCloudProcessor(Node):
    def __init__(self):
        super().__init__('posture_estimation')
        self.br = TransformBroadcaster(self)
        self.pub = self.create_publisher(TransformStamped, '/estimated_transform', 10)
        
        # 可視化用パブリッシャー
        self.downsampled_pub = self.create_publisher(PointCloud2, '/downsampled_points', 10)
        self.model_pub = self.create_publisher(PointCloud2, '/model_points', 10)
        self.source_edge_pub = self.create_publisher(PointCloud2, '/source_edge_points', 10)
        self.template_edge_pub = self.create_publisher(PointCloud2, '/template_edge_points', 10) 
        self.aligned_model_pub = self.create_publisher(PointCloud2, '/aligned_model_points', 10)
        
        self.frame_id = "camera_depth_optical_frame"
        self.model_ref_frame = "model_reference" 
        self.model_ref_pub_timer = self.create_timer(0.5, self.publish_model_ref_tf) 
        
        self.voxel_size = 0.005
        self.pattern = 'b' # 'b': ball, 'c': chipstar, etc.
        self.measured_pcd = None
        
        # モデル点群の読み込み
        model_dir = "ModelPCD"
        model_filename = "scissor_model.pcd" # 必要に応じて "pen_model.pcd" などに変更
        model_path = os.path.join(model_dir, model_filename)

        if os.path.exists(model_path):
            raw_model_pcd = o3d.io.read_point_cloud(model_path)
            self.model_pcd = raw_model_pcd.voxel_down_sample(voxel_size=self.voxel_size)
            self.get_logger().info(f"Loaded model: {len(self.model_pcd.points)} points.")
        else:
            self.get_logger().error(f"Model file {model_path} not found!")

        self.model_publish_timer = self.create_timer(1.0, self.model_timer_callback) 
        self.sub = self.create_subscription(PointCloud2, '/filtered_points', self.point_cloud_callback, 10)
        
    def point_cloud_callback(self, msg):
        points_iterator = point_cloud2.read_points(msg, field_names=("x", "y", "z", "rgb"), skip_nans=True)
        points_structured = np.array(list(points_iterator))
        
        if points_structured.shape[0] == 0:
            return

        xyz = np.vstack([points_structured['x'], points_structured['y'], points_structured['z']]).T.astype(np.float64)
        self.measured_pcd = create_pcd(xyz)

        if 'rgb' in points_structured.dtype.names:
            rgb_int = points_structured['rgb'].view(np.uint32) 
            r = ((rgb_int >> 16) & 0x0000ff).astype(np.float64) / 255.0
            g = ((rgb_int >> 8) & 0x0000ff).astype(np.float64) / 255.0
            b = (rgb_int & 0x0000ff).astype(np.float64) / 255.0
            self.measured_pcd.colors = o3d.utility.Vector3dVector(np.vstack([r, g, b]).T)

        try:
            self.process_point_cloud()
        except Exception as e:
            self.get_logger().error(f"Error in process_point_cloud: {e}", throttle_duration_sec=1.0)

    def process_point_cloud(self):
        start = time.time()
        if self.measured_pcd is None: return

        downsampled_measured_pcd = self.measured_pcd.voxel_down_sample(voxel_size=self.voxel_size)
        if len(downsampled_measured_pcd.points) == 0: return

        self.publish_pcd_to_topic(downsampled_measured_pcd, self.downsampled_pub, "/downsampled_points")

        def pcd_to_numpy(pcd):
            pts = np.asarray(pcd.points)
            if pcd.has_colors():
                clr = np.asarray(pcd.colors)
                return np.hstack([pts, clr])
            return pts

        numpy_model = pcd_to_numpy(self.model_pcd)
        numpy_measured = pcd_to_numpy(downsampled_measured_pcd)

        registration_model = Registration.Registration(pattern=self.pattern)
        result = registration_model.register(numpy_model, numpy_measured)
        
        if not isinstance(result, dict) or 'est_T' not in result:
            self.get_logger().error("Registration failed.")
            return

        est_T = result['est_T']
        if est_T is not None:
            T_final_tf = np.linalg.inv(est_T) 
            transformed_model = copy.deepcopy(self.model_pcd)
            transformed_model.transform(T_final_tf) 
            self.publish_pcd_to_topic(transformed_model, self.aligned_model_pub, "/aligned_model_points")
            self.publish_transform(T_final_tf)
        
        self.publish_visualization_pcds(registration_model)
        self.get_logger().info(f'Total time [s] = {time.time() - start:.4f}')

    def publish_model_ref_tf(self):
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = self.frame_id
        t.child_frame_id = self.model_ref_frame
        t.transform.rotation.w = 1.0
        self.br.sendTransform(t)

    def model_timer_callback(self):
        if self.model_pcd is not None:
            self.publish_pcd_to_topic(self.model_pcd, self.model_pub, "/model_points", frame_id=self.model_ref_frame)

    def publish_visualization_pcds(self, registration_model):
        vis_msgs = registration_model.get_visualization_messages(frame_id=self.frame_id)
        if 'source_edge_pcd' in vis_msgs and vis_msgs['source_edge_pcd']:
            self.source_edge_pub.publish(vis_msgs['source_edge_pcd'])
        if 'template_edge_pcd' in vis_msgs and vis_msgs['template_edge_pcd']:
            self.template_edge_pub.publish(vis_msgs['template_edge_pcd'])

    def publish_pcd_to_topic(self, o3d_pcd, publisher, topic_name, frame_id=None):
        if frame_id is None: frame_id = self.frame_id
        points = np.asarray(o3d_pcd.points)
        if points.shape[0] == 0: return

        colors = np.asarray(o3d_pcd.colors) if o3d_pcd.has_colors() else np.zeros_like(points)
        rgb_combined = ((colors[:, 0]*255).astype(np.uint32) << 16) | \
                       ((colors[:, 1]*255).astype(np.uint32) << 8) | \
                       ((colors[:, 2]*255).astype(np.uint32))

        data = np.empty(points.shape[0], dtype=[('x', np.float32), ('y', np.float32), ('z', np.float32), ('rgb', np.uint32)])
        data['x'], data['y'], data['z'], data['rgb'] = points[:, 0], points[:, 1], points[:, 2], rgb_combined

        fields = [
            point_cloud2.PointField(name='x', offset=0, datatype=point_cloud2.PointField.FLOAT32, count=1),
            point_cloud2.PointField(name='y', offset=4, datatype=point_cloud2.PointField.FLOAT32, count=1),
            point_cloud2.PointField(name='z', offset=8, datatype=point_cloud2.PointField.FLOAT32, count=1),
            point_cloud2.PointField(name='rgb', offset=12, datatype=point_cloud2.PointField.UINT32, count=1), 
        ]
        header = Header(stamp=self.get_clock().now().to_msg(), frame_id=frame_id)
        pc_msg = point_cloud2.create_cloud(header, fields, data)
        publisher.publish(pc_msg)

    def publish_transform(self, T_final_tf):
        R_z_180 = np.eye(4)
        R_z_180[0,0], R_z_180[1,1] = -1, -1
        R_y_neg90 = np.array([[0,0,-1,0], [0,1,0,0], [1,0,0,0], [0,0,0,1]])
        T_corrected = T_final_tf @ R_z_180 @ R_y_neg90

        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = self.frame_id
        t.child_frame_id = "Posture_of_object"
        t.transform.translation.x, t.transform.translation.y, t.transform.translation.z = T_corrected[:3, 3]
        
        q = tf_transformations.quaternion_from_matrix(T_corrected)
        t.transform.rotation.x, t.transform.rotation.y, t.transform.rotation.z, t.transform.rotation.w = q

        # クォータニオン情報の表示
        self.get_logger().info(f"Estimated Quaternion: x={q[0]:.4f}, y={q[1]:.4f}, z={q[2]:.4f}, w={q[3]:.4f}")

        self.pub.publish(t)
        self.br.sendTransform(t)

def main(args=None):
    rclpy.init(args=args)
    processor = PointCloudProcessor()
    rclpy.spin(processor)
    processor.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()