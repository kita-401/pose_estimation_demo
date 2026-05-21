import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2, CameraInfo, PointField
from cv_bridge import CvBridge
import message_filters
from ultralytics import YOLO
import numpy as np
import cv2
from sensor_msgs_py import point_cloud2
import torch  # GPU判定に必要
import time   # 処理時間計測に必要
import os

class Yolo26FilteredNode(Node):
    def __init__(self):
        super().__init__('yolo26_filtered_node')
        self.bridge = CvBridge()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # スクリプトがあるディレクトリ（.../demo/scripts_GPU）の絶対パスを取得
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # YOLO_model/yolo26s-seg.pt のフルパスを生成
        model_path = os.path.join(script_dir, "YOLO_model", "yolo26s-seg.pt")
        
        # ファイルの存在チェックをしてからロード（エラー対策）
        if os.path.exists(model_path):
            self.model = YOLO(model_path).to(self.device)
            self.get_logger().info(f"Loaded YOLO model from: {model_path}")
        else:
            self.get_logger().error(f"YOLO model file NOT found at: {model_path}")
            # 万が一見つからない場合はカレントディレクトリ等から探すフォールバック（旧処理）
            self.model = YOLO('yolo26s-seg.pt').to(self.device)

        self.intrinsics = None
        
        # --- サブスクライバの設定 ---
        self.info_sub = self.create_subscription(
            CameraInfo, 
            '/camera/camera/aligned_depth_to_color/camera_info', 
            self.info_callback, 10
        )
        self.rgb_sub = message_filters.Subscriber(self, Image, '/camera/camera/color/image_rect_raw')
        self.pc_sub = message_filters.Subscriber(self, PointCloud2, '/camera/camera/depth/color/points')
        
        # 画像と点群の時刻同期 (ApproximateTimeSynchronizer)
        self.ts = message_filters.ApproximateTimeSynchronizer([self.rgb_sub, self.pc_sub], 10, 0.1)
        self.ts.registerCallback(self.image_callback)

        # --- パブリッシャの設定 ---
        self.filtered_pc_pub = self.create_publisher(PointCloud2, '/filtered_points', 10)
        self.image_pub = self.create_publisher(Image, '/yolo_annotated_image', 10)

        self.get_logger().info(f"YOLO Node: Running on {self.device.upper()} mode")

    def info_callback(self, msg):
        self.intrinsics = msg
        self.get_logger().info("Camera intrinsics received.")
        self.destroy_subscription(self.info_sub)

    def image_callback(self, rgb_msg, pc_msg): 
        if self.intrinsics is None: return

        # 処理開始時間を記録
        start_time = time.time()

        try:
            # ROSメッセージをOpenCV形式に変換
            cv_image = self.bridge.imgmsg_to_cv2(rgb_msg, "bgr8")
            h, w = cv_image.shape[:2]
            
            # PointCloud2データをNumPy配列として展開
            raw_data = np.frombuffer(pc_msg.data, dtype=np.uint8).reshape(h, w, pc_msg.point_step)
            # x, y, z の3つのfloat32（各4byte）を抽出
            xyz_array = raw_data[:, :, 0:12].view(dtype=np.float32).reshape(h, w, 3)
        except Exception as e:
            self.get_logger().error(f"Data processing error: {e}")
            return

        # --- YOLO推論の実行 (GPU/CPU自動切替) ---
        results = self.model.predict(cv_image, verbose=False, device=self.device, conf=0.25)

        annotated_frame = cv_image.copy()

        # 物体が検出され、セグメンテーションマスクがある場合のみ処理
        if results[0].masks is not None:
            masks = results[0].masks.data
            
            # 最初に見つかった物体 (Index 0) のみを抽出
            mask_raw = masks[0].cpu().numpy()
            m_final = cv2.resize(mask_raw, (w, h))
            m_bool = m_final > 0.5

            # マスク範囲内の点群を抽出
            final_points = xyz_array[m_bool]
            
            # 有効な点（NaN除外、距離0.1m〜2.0m以内）をフィルタリング
            valid_mask = ~np.isnan(final_points).any(axis=1) & (final_points[:, 2] > 0.1) & (final_points[:, 2] < 2.0)
            final_points = final_points[valid_mask]

            if len(final_points) > 50:
                # pose_estimate.py が要求する rgb フィールド付きでパブリッシュ
                msg = self.create_pc2_msg(final_points, rgb_msg.header)
                self.filtered_pc_pub.publish(msg)
                
                # デバッグ表示用の描画
                box = results[0].boxes.xyxy[0].cpu().numpy().astype(int)
                cv2.rectangle(annotated_frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
                cv2.putText(annotated_frame, f"Detected: {len(final_points)} pts", (box[0], box[1] - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # アノテーション済み画像をパブリッシュ
        self.image_pub.publish(self.bridge.cv2_to_imgmsg(annotated_frame, "bgr8"))

        # 処理時間の計算と出力 (1秒ごとに間引く)
        end_time = time.time()
        self.get_logger().info(f'YOLO Segmentation Total Time: {end_time - start_time:.4f} [s]', throttle_duration_sec=1.0)

    def create_pc2_msg(self, points, header):
        """pose_estimate.py との互換性を保つため rgb フィールドを含む PointCloud2 を作成"""
        msg = PointCloud2()
        msg.header = header
        msg.header.frame_id = "camera_depth_optical_frame"
        msg.height = 1
        msg.width = len(points)

        # フィールド定義 (x, y, z, rgb)
        msg.fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='rgb', offset=12, datatype=PointField.UINT32, count=1), 
        ]
        
        msg.is_bigendian = False
        msg.point_step = 16  # 4 fields * 4 bytes
        msg.row_step = msg.point_step * len(points)
        msg.is_dense = True

        # 点群データとダミーの色情報（白）を結合
        data_to_publish = np.empty(len(points), dtype=[
            ('x', np.float32), 
            ('y', np.float32), 
            ('z', np.float32), 
            ('rgb', np.uint32)
        ])
        data_to_publish['x'] = points[:, 0].astype(np.float32)
        data_to_publish['y'] = points[:, 1].astype(np.float32)
        data_to_publish['z'] = points[:, 2].astype(np.float32)
        data_to_publish['rgb'] = 16777215  # 0xFFFFFF (White)

        msg.data = data_to_publish.tobytes()
        return msg

def main(args=None):
    rclpy.init(args=args)
    node = Yolo26FilteredNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()