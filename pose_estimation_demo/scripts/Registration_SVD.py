import numpy as np
import torch
import open3d as o3d

class SVD_ICP:
    def __init__(self, threshold=0.01, max_iteration=30): 
        self.threshold = threshold 
        self.criteria = o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iteration) 
        # CUDAが利用可能ならGPUを使用
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.template_mean_t = None
        self.source_mean_t = None
        self.template_pcd_for_icp = None 
        self.source_pcd_for_icp = None   

    def preprocess(self, template, source):
        # データをGPUに転送
        if not torch.is_tensor(template):
            template_t = torch.from_numpy(template[:, :3]).to(self.device).float()
            source_t = torch.from_numpy(source[:, :3]).to(self.device).float()
        else:
            template_t = template[:, :3].to(self.device).float()
            source_t = source[:, :3].to(self.device).float()

        self.template_mean_t = torch.mean(template_t, dim=0, keepdim=True)
        self.source_mean_t = torch.mean(source_t, dim=0, keepdim=True)
        
        self.template_pcd_for_icp = template_t - self.template_mean_t
        self.source_pcd_for_icp = source_t - self.source_mean_t

    def compute_fitness_gpu(self, source_points, target_points, threshold):
        """
        GPU上で最近傍距離を計算し、Fitnessを算出
        """
        # source_points: [N, 3], target_points: [M, 3]
        # メモリ効率のため、チャンクに分けるか、小規模なら一括で距離行列を計算
        # ここではシンプルな実装（点数が多い場合は注意）
        dist_sq = torch.cdist(source_points, target_points, p=2)
        min_dist, _ = torch.min(dist_sq, dim=1)
        inliers = torch.sum(min_dist < threshold)
        return inliers.float() / source_points.shape[0]

    def __call__(self, template, source, pattern=None):
        self.preprocess(template, source)
        
        # --- 1. SVD (GPU) ---
        def get_basis(p):
            _, _, Vt = torch.linalg.svd(p, full_matrices=False)
            return Vt.T # [V1, V2, V3]

        basis_t = get_basis(self.template_pcd_for_icp) # ターゲットの基底
        basis_s = get_basis(self.source_pcd_for_icp)   # ソースの基底
        
        # 軸反転パターンの定義 (1: 反転なし, -1: 反転)
        # SVDの軸の向きは任意なので、全パターンの組み合わせを確認
        flip_patterns = [
            [ 1,  1,  1],
            [-1,  1,  1], # X反転
            [ 1, -1,  1], # Y反転
            [ 1,  1, -1], # Z反転
            [-1, -1,  1],
            [-1,  1, -1],
            [ 1, -1, -1],
            [-1, -1, -1]
        ]

        best_fit = -1.0
        best_R = None
        check_threshold = 0.005 # 5mm

        for p in flip_patterns:
            # 符号を適用したソース側の基底を作成
            flipped_basis_s = basis_s * torch.tensor(p, device=self.device)
            
            # 回転行列 R = basis_t @ flipped_basis_s.T
            R_candidate = torch.mm(basis_t, flipped_basis_s.T)
            
            # 鏡像反転の補正 (det(R) < 0 の場合は正しい回転行列ではないためスキップ or 補正)
            # 物理的にあり得る姿勢 (det=1) のみを評価対象にする
            if torch.linalg.det(R_candidate) < 0:
                continue

            # Fitness計測
            transformed_source = torch.mm(self.source_pcd_for_icp, R_candidate.T)
            fit = self.compute_fitness_gpu(transformed_source, self.template_pcd_for_icp, check_threshold)

            if fit > best_fit:
                best_fit = fit
                best_R = R_candidate

        # 全パターン評価しても R が決まらなかった場合のフォールバック
        if best_R is None:
            best_R = torch.mm(basis_t, basis_s.T)

        # --- 2. ICP (Open3D) ---
        T_init = np.eye(4)
        T_init[:3, :3] = best_R.detach().cpu().numpy()
        
        source_o3d = o3d.geometry.PointCloud()
        source_o3d.points = o3d.utility.Vector3dVector(self.source_pcd_for_icp.detach().cpu().numpy())
        target_o3d = o3d.geometry.PointCloud()
        target_o3d.points = o3d.utility.Vector3dVector(self.template_pcd_for_icp.detach().cpu().numpy())
        
        reg = o3d.pipelines.registration.registration_icp(
            source_o3d, target_o3d, self.threshold, T_init,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(), self.criteria)

        R_final = reg.transformation[:3, :3]
        t_final = reg.transformation[:3, 3] + \
                  self.template_mean_t.cpu().numpy().flatten() - \
                  (R_final @ self.source_mean_t.cpu().numpy().flatten())
        
        T_final = np.eye(4)
        T_final[:3, :3] = R_final
        T_final[:3, 3] = t_final

        return {'est_R': R_final, 'est_t': t_final, 'est_T': T_final}

    def get_visualization_messages(self, frame_id="camera_link"):
        return {}

class Registration:
    def __init__(self, pattern="A"):
        self.reg_algorithm = SVD_ICP()
        self.pattern = pattern
    def register(self, template, source):
        return self.reg_algorithm(template, source, self.pattern)
    def get_visualization_messages(self, frame_id="camera_depth_optical_frame"):
        return self.reg_algorithm.get_visualization_messages(frame_id)