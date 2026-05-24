import numpy as np

class APFController:
    def __init__(self, dh_params, ka=1.0, dls_lambda=0.1):
        """
        初始化 APF 控制器。
        :param dh_params: UR5 的 DH 参数
        :param ka: 引力增益
        :param dls_lambda: 阻尼最小二乘法 (DLS) 的阻尼因子
        """
        self.dh_params = dh_params
        self.Ka = ka
        self.dls_lambda = dls_lambda

    def get_ee_pose(self, q):
        """
        利用 DH 参数计算正运动学 (FK)，获取末端执行器 (EE) 的笛卡尔坐标。
        """
        T = np.eye(4)
        for i in range(6):
            d = self.dh_params['d'][i]
            a = self.dh_params['a'][i]
            alpha = self.dh_params['alpha'][i]
            theta = q[i]
            
            ct, st = np.cos(theta), np.sin(theta)
            ca, sa = np.cos(alpha), np.sin(alpha)
            
            Ti = np.array([
                [ct, -st*ca,  st*sa, a*ct],
                [st,  ct*ca, -ct*sa, a*st],
                [0,   sa,     ca,    d],
                [0,   0,      0,     1]
            ])
            T = T @ Ti
            
        return T[:3, 3] # 返回末端执行器的 [x, y, z]

    def get_jacobian(self, q):
        """
        计算几何雅可比矩阵 (Geometric Jacobian) 的位置部分 (3x6)。
        """
        T = np.eye(4)
        origins = [T[:3, 3]]
        z_axes = [T[:3, 2]]
        
        for i in range(6):
            d = self.dh_params['d'][i]
            a = self.dh_params['a'][i]
            alpha = self.dh_params['alpha'][i]
            theta = q[i]
            
            ct, st = np.cos(theta), np.sin(theta)
            ca, sa = np.cos(alpha), np.sin(alpha)
            
            Ti = np.array([
                [ct, -st*ca,  st*sa, a*ct],
                [st,  ct*ca, -ct*sa, a*st],
                [0,   sa,     ca,    d],
                [0,   0,      0,     1]
            ])
            T = T @ Ti
            origins.append(T[:3, 3])
            z_axes.append(T[:3, 2])
            
        ee_pos = origins[-1]
        J_pos = np.zeros((3, 6))
        
        for i in range(6):
            J_pos[:, i] = np.cross(z_axes[i], (ee_pos - origins[i]))
            
        return J_pos

    def compute_joint_increment(self, current_q, target_pos):
        """
        计算引力场导致的关节增量。
        公式: Δq = J_pinv * Ka * (P_target - P_ee)
        """
        current_pos = self.get_ee_pose(current_q)
        cartesian_error = target_pos - current_pos
        
        J = self.get_jacobian(current_q)
        
        # 使用阻尼最小二乘法 (DLS) 求伪逆
        J_pinv = J.T @ np.linalg.inv(J @ J.T + (self.dls_lambda**2) * np.eye(3))
        
        delta_q = J_pinv @ (self.Ka * cartesian_error)
        
        return delta_q
