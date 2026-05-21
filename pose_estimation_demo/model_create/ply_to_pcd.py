import open3d as o3d

# PLYファイルの読み込み
pcd = o3d.io.read_point_cloud("TF_setting_penv2.ply")

# 点群データを表示
o3d.visualization.draw_geometries([pcd])

# 点群データをPCD形式で保存
o3d.io.write_point_cloud("scissor_penv2.pcd", pcd)

