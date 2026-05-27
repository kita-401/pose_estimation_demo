import os
import sys
import copy
import time
import numpy as np
import open3d as o3d
import asyncio
from scipy.spatial.transform import Rotation as R
import tf_transformations  # または transformations

from std_msgs.msg import String
from rclpy.node import Node
from type.agent_state import AgentState
from core.base import Base

# 姿勢推定のコアアルゴリズム
from pose_estimation_demo.scripts import Registration_SVD as Registration

class PoseEstimationNode(Node, Base):
    def __init__(self):
        super().__init__('pose_estimation_node')

        # UI通知用パブリッシャー
        self.status_stream_pub = self.create_publisher(String, '/chatbot_response_stream', 10)
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # クラス内で最新の測定点群（measured_pcd）を保持する（コールバック等から更新される前提）
        self.measured_pcd = None 

    async def handle_pose_estimation_node(self, state: AgentState) -> AgentState:
        """
        [エージェント用ノード] 実際の点群から計算された本物のクォータニオンを返す
        """
        detected_object = state.get("detected_object", "scissor")
        voxel_size = state.get("voxel_size", 0.005)
        pattern = state.get("pattern", "b")

        model_mapping = {
            "scissor": "scissor_model.pcd",
            "pen": "pen_modelv2.pcd",
            "pipe": "T_joint_pipe_10000_half_model.pcd"
        }
        
        model_filename = model_mapping.get(detected_object, "scissor_model.pcd")
        model_path = os.path.join(self.script_dir, "ModelPCD", model_filename)

        # 1. カメラ点群の存在チェック
        if self.measured_pcd is None or len(self.measured_pcd.points) == 0:
            error_msg = "\n❌ エラー: カメラ点群（measured_pcd）がまだ取得できていません。"
            self.status_stream_pub.publish(String(data=error_msg))
            return {"text_response": error_msg, "pose_estimation_status": "failed"}

        if not os.path.exists(model_path):
            error_msg = f"\n❌ エラー: モデルファイルが見つかりません: {model_filename}"
            self.status_stream_pub.publish(String(data=error_msg))
            return {"text_response": error_msg, "pose_estimation_status": "failed"}

        start_msg = f"🤖 [姿勢推定] {detected_object} の本物の位置姿勢を計算中..."
        self.status_stream_pub.publish(String(data=start_msg))

        try:
            # 2. 【本物の計算】スレッドプールでOpen3D/SVDの重い計算をノンブロッキング実行
            result = await asyncio.to_thread(
                self._execute_registration, model_path, voxel_size, pattern
            )
            
            if result and result.get("success", False):
                q = result["quaternion"]
                t = result["translation"]
                
                success_msg = (
                    f"\n✅ 姿勢推定に成功（リアルタイム計測値）"
                    f"\n位置(x,y,z): [{t[0]:.4f}, {t[1]:.4f}, {t[2]:.4f}]"
                    f"\n姿勢(x,y,z,w): [{q[0]:.4f}, {q[1]:.4f}, {q[2]:.4f}, {q[3]:.4f}]"
                )
                self.status_stream_pub.publish(String(data=success_msg))
                
                # 3. 本物の計算結果をAgentStateに詰めて返す
                return {
                    "text_response": success_msg,
                    "pose_estimation_status": "success",
                    "estimated_quaternion": {"x": q[0], "y": q[1], "z": q[2], "w": q[3]},
                    "estimated_translation": {"x": t[0], "y": t[1], "z": t[2]},
                    "chat_history": self._add_history(state, success_msg)
                }
            else:
                fail_msg = "\n❌ 姿勢推定に失敗しました（レジストレーション不一致）。"
                self.status_stream_pub.publish(String(data=fail_msg))
                return {"text_response": fail_msg, "pose_estimation_status": "failed"}

        except Exception as e:
            err_msg = f"\n❌ 推定プロセス中に計算エラーが発生: {str(e)}"
            self.status_stream_pub.publish(String(data=err_msg))
            return {"text_response": err_msg, "pose_estimation_status": "error"}

    def _execute_registration(self, model_path, voxel_size, pattern):
        """
        【完全リアルタイム計算】
        元の pose_estimate.py のロジックに従い、現在の点群データからクォータニオンを抽出する
        """
        # 1. モデル点群の読み込みとダウンサンプリング
        raw_model_pcd = o3d.io.read_point_cloud(model_path)
        model_pcd = raw_model_pcd.voxel_down_sample(voxel_size=voxel_size)

        # 2. 測定点群（カメラデータ）のダウンサンプリング
        downsampled_measured_pcd = self.measured_pcd.voxel_down_sample(voxel_size=voxel_size)
        if len(downsampled_measured_pcd.points) == 0:
            return {"success": False}

        # 3. Numpy変換関数
        def pcd_to_numpy(pcd):
            pts = np.asarray(pcd.points)
            if pcd.has_colors():
                clr = np.asarray(pcd.colors)
                return np.hstack([pts, clr])
            return pts

        numpy_model = pcd_to_numpy(model_pcd)
        numpy_measured = pcd_to_numpy(downsampled_measured_pcd)

        # 4. レジストレーション（SVDマッチング）の実行
        registration_model = Registration.Registration(pattern=pattern)
        result = registration_model.register(numpy_model, numpy_measured)
        
        if not isinstance(result, dict) or 'est_T' not in result:
            return {"success": False}

        est_T = result['est_T']
        if est_T is None:
            return {"success": False}

        # 5. 元コード通りの座標系反転・補正（逆行列 ＆ 軸回転調整）
        T_final_tf = np.linalg.inv(est_T) 
        
        R_z_180 = np.eye(4)
        R_z_180[0,0], R_z_180[1,1] = -1, -1
        R_y_neg90 = np.array([[0,0,-1,0], [0,1,0,0], [1,0,0,0], [0,0,0,1]])
        
        # 最終的な補正済み同次変換行列
        T_corrected = T_final_tf @ R_z_180 @ R_y_neg90

        # 6.変換行列から本物のクォータニオンと並進ベクトルを抽出
        q = tf_transformations.quaternion_from_matrix(T_corrected)  # [x, y, z, w] の numpy 配列
        t = T_corrected[:3, 3]                                      # [x, y, z] の並進ベクトル

        return {
            "success": True,
            "quaternion": q.tolist(),
            "translation": t.tolist()
        }