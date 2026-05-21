# pose_estimation_demo
デモのためのSVDを用いた三次元姿勢推定手法です．

# 3D Object Pose Estimation System using YOLOv26 & SVD-ICP
Intel RealSense D405 から取得したRGB-Dデータに対し、YOLOによるインスタンスセグメンテーション（2D）と、SVD（特異値分解）およびICP（反復最近傍点）アルゴリズム（3D）を融合して物体の3D姿勢を高速に推定するROS 2パッケージです。
SVD演算および近傍探索をPyTorchを用いてGPU並列化することで、リアルタイムな位置合わせを実現しています。

## 概要 (System Overview)

本システムは以下の2つのメインノードから構成されます。

1. **Yolo26FilteredNode (`Yolov26_seg.py`)**
   - **RealSense D405** のカラー画像からターゲット（ハサミ、ペン、T字パイプなど）をセグメンテーションします。
   - 同期された `PointCloud2` から、マスク領域内かつ指定距離内（0.1m〜2.0m）の有効な点群のみを高速に抽出して配信します。
2. **PointCloudProcessor (`pose_estimate.py`)**
   - 抽出された点群と、あらかじめ用意した対象物の3Dモデル点群（`.pcd`）とのマッチングを行います。
   - **`Registration_SVD.py`** を用いてSVDによる軸反転パターンの全探索およびOpen3Dによる精緻なICP調整を実行し、物体の3D姿勢（位置・クォータニオン）をTF (`Posture_of_object`) としてリアルタイム配信します。

---

## 対象オブジェクトと3Dモデル作成
本システムは、**ハサミ (scissor)**、**ペン (pen)**、**T字パイプ (t_pipe)** などのオブジェクトに対応しています。

**新たにモデル点群を作成したい場合は，以下の手順に沿って作成してください．**
１．三次元計測センサの起動（realsenseなど）
２．後述するYOLOノードを起動
３．`model_create/` フォルダにある **pointcloud_saver.py**を実行（モデル点群の名前は，pointcloud_saver.py内を適宜修正してください．）
４．パッケージ内の `model_create/` フォルダにある **Blenderファイル** を使用して対象物の形状データを編集・エクスポートし，モデルPCD（`.ply` ファイル）を作成
５．`model_create/` フォルダにある**ply_to_pcd.py**を使用して，`.pcd` ファイル）を作成(ply_to_pcd.py内の点群名は適宜変更してください)
---

## 環境要件 & 依存パッケージ (Prerequisites)

### 1. OS / ミドルウェア
- Ubuntu 22.04+
- ROS 2 Humble+
- **Intel RealSense ROS ラッパー** (`realsense2_camera`)

### 2. 必要なPythonライブラリ / ROS拡張
GPU（CUDA）環境を推奨します。以下のコマンドで必要なライブラリをインストールしてください。

```bash
# 主要なPythonライブラリのインストール
pip install ultralytics open3d numpy torch opencv-python scipy

# 座標変換・クォータニオン計算に必要な拡張ライブラリ
pip install transformations
```

### 3. ビルドの実行
```bash
colcon build --packages-select point_free_occlusion_pose
source install/setup.bash
source /opt/ros/humble/setup.bash
```

### 4. 物体認識および姿勢推定の実行

```bash
# yoloの実行（realsenseD405に対応）
ros2 run pose_estimation Yolov26_seg
```

```bash
# 姿勢推定の実行
ros2 run pose_estimation pose_estimate
```