import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
import numpy as np
import struct
import os
import math # math.isfinite の代わりに np.isfinite を使用しているため、ここでは不要だが念のため残す

class PointCloudSaver(Node):
    """
    /filtered_pointsトピックのPointCloud2メッセージをPLYファイルとして保存するノード。
    
    メッセージを一つ受信後、PLYファイルに保存し、ノードを終了する。
    """
    def __init__(self, output_filename="chipstar_model.ply"):
        super().__init__('pointcloud_saver_node')
        self.output_filename = output_filename
        
        # サブスクライバーの作成
        self.subscription = self.create_subscription(
            PointCloud2,
            '/filtered_points',
            self.listener_callback,
            1  # QoS設定: 1は最新のメッセージを1つだけキューに保持する設定
        )
        self.get_logger().info(f'Subscribing to /filtered_points and ready to save to {self.output_filename}')
        self.saved = False

    def listener_callback(self, msg: PointCloud2):
        if self.saved:
            return

        self.get_logger().info('Received a PointCloud2 message. Starting conversion...')
        
        # PointCloud2メッセージをNumPy配列に変換する処理
        points_np = self.convert_pointcloud2_to_numpy(msg)

        if points_np is None or points_np.size == 0:
            self.get_logger().warn('Converted point cloud is empty or invalid. Skipping save and waiting for next message.')
            return

        # PLYファイルとして保存
        self.save_to_ply(points_np)
        self.saved = True
        
        self.get_logger().info(f'Successfully saved {points_np.shape[0]} points to {self.output_filename}')
        
        # 保存後、ノードを終了
        self.get_logger().info('Stopping node after saving.')
        self.destroy_node()


    def convert_pointcloud2_to_numpy(self, msg: PointCloud2):
        """
        PointCloud2メッセージのバイナリデータをNumPy配列に変換する
        
        PointCloud2のフィールド情報（オフセットとデータ型）に基づいて、
        正確に各点のデータを読み取ります。
        """
        
        # PointCloud2のDataTypeとPythonのstructフォーマットの対応表
        FIELD_DATA_TYPE = {
            1: 'b',  2: 'B',  3: 'h',  4: 'H',  # 整数系
            5: 'i',  6: 'I',  7: 'f',  8: 'd',  # 浮動小数点/その他
        }
        
        fields_info = {f.name: (f.offset, f.datatype) for f in msg.fields}
        
        # 必須フィールドの確認
        if 'x' not in fields_info or 'y' not in fields_info or 'z' not in fields_info:
            self.get_logger().error("PointCloud2 message is missing x, y, or z fields.")
            return None
        
        buffer = msg.data
        point_step = msg.point_step
        num_points = msg.width * msg.height
        # エンディアン：リトルエンディアン '<' (false) or ビッグエンディアン '>' (true)
        endian_char = '<' if not msg.is_bigendian else '>' 
        
        points_list = []
        
        # RGB関連の準備
        has_rgb_field = 'rgb' in fields_info
        
        # RGBフィールドが存在する場合、常に UINT32 として読み取るための設定
        if has_rgb_field:
            offset_rgb = fields_info['rgb'][0]
            size_rgb = 4 
            rgb_struct_format = 'I' # UINT32
        
        for i in range(num_points):
            point_start = i * point_step
            
            # X, Y, Z, R, G, Bの初期値
            x, y, z = np.nan, np.nan, np.nan
            r, g, b = 255, 255, 255 
            
            # 1. X, Y, Zの抽出（各フィールドのオフセットを信頼して個別に読み込む）
            try:
                # X
                offset_x, dtype_x = fields_info['x']
                size_x = struct.calcsize(FIELD_DATA_TYPE.get(dtype_x, 'f'))
                x_data = buffer[point_start + offset_x : point_start + offset_x + size_x]
                x, = struct.unpack(f'{endian_char}{FIELD_DATA_TYPE.get(dtype_x, "f")}', x_data)

                # Y
                offset_y, dtype_y = fields_info['y']
                size_y = struct.calcsize(FIELD_DATA_TYPE.get(dtype_y, 'f'))
                y_data = buffer[point_start + offset_y : point_start + offset_y + size_y]
                y, = struct.unpack(f'{endian_char}{FIELD_DATA_TYPE.get(dtype_y, "f")}', y_data)

                # Z
                offset_z, dtype_z = fields_info['z']
                size_z = struct.calcsize(FIELD_DATA_TYPE.get(dtype_z, 'f'))
                z_data = buffer[point_start + offset_z : point_start + offset_z + size_z]
                z, = struct.unpack(f'{endian_char}{FIELD_DATA_TYPE.get(dtype_z, "f")}', z_data)

            except (struct.error, KeyError, IndexError):
                # 座標値の読み取りに失敗した場合、この点をスキップ
                continue
            
            # 2. RGBの抽出
            if has_rgb_field:
                try:
                    # RGBがパックされたUINT32形式
                    rgb_data = buffer[point_start + offset_rgb : point_start + offset_rgb + size_rgb]
                    
                    # UINT32としてアンパック
                    rgb_packed, = struct.unpack(f'{endian_char}{rgb_struct_format}', rgb_data)
                    
                    # floatとして誤読された場合に備え、int()で整数に変換してビットシフトエラーを回避
                    rgb_int = int(rgb_packed) 
                    
                    # UINT32からR, G, Bを抽出
                    r = (rgb_int >> 16) & 0xFF
                    g = (rgb_int >> 8) & 0xFF
                    b = rgb_int & 0xFF

                except (struct.error, ValueError, TypeError, IndexError):
                    # 読み取り失敗時や型変換エラー時、デフォルトの白(255, 255, 255)を使用
                    pass

            points_list.append((x, y, z, r, g, b))

        # NumPy structured arrayとして結合
        dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), 
                 ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
        
        if not points_list:
            return np.array([], dtype=dtype)
            
        points_np = np.array(points_list, dtype=dtype)
        
        # NaN（無効な点）を除外
        # NumPyの浮動小数点数の有限性チェックを使用
        valid_points = np.logical_and.reduce((np.isfinite(points_np['x']), 
                                              np.isfinite(points_np['y']), 
                                              np.isfinite(points_np['z'])))
        return points_np[valid_points]


    def save_to_ply(self, points_np):
        """
        NumPy配列をASCII形式のPLYファイルとして保存する
        """
        header = [
            'ply',
            'format ascii 1.0',
            f'element vertex {points_np.shape[0]}',
            'property float x',
            'property float y',
            'property float z',
            'property uchar red',
            'property uchar green',
            'property uchar blue',
            'end_header'
        ]

        # データをヘッダーの後に追加
        with open(self.output_filename, 'w') as f:
            f.write('\n'.join(header) + '\n')
            
            # データを整形して書き込み
            for point in points_np:
                # 座標は6桁の小数点以下、色は整数として書き込む
                f.write(f"{point['x']:.6f} {point['y']:.6f} {point['z']:.6f} {point['red']} {point['green']} {point['blue']}\n")

def main(args=None):
    # ノードを生成するための初期化
    rclpy.init(args=args)
    
    pointcloud_saver = PointCloudSaver()

    # スピンしてコールバックを待つ
    try:
        rclpy.spin(pointcloud_saver)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        pointcloud_saver.get_logger().error(f"Critical error occurred: {e}")
    finally:
        # シャットダウン処理
        if rclpy.ok():
            # ノードがまだ破棄されていない場合、ここで破棄
            if pointcloud_saver.saved is False and rclpy.ok():
                 pointcloud_saver.destroy_node()
            rclpy.shutdown()

if __name__ == '__main__':
    main()
