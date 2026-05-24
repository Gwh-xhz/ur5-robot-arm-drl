import gymnasium as gym
from gymnasium import spaces
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.client import Client
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration
from rclpy.qos import QoSProfile, ReliabilityPolicy
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from geometry_msgs.msg import TransformStamped, Point
import time
import threading
# 强制使用 Gazebo 原生的 ContactsState 消息类型
from gazebo_msgs.msg import ContactsState
# 导入自定义服务接口
from ur_simulation_gazebo.srv import GenerateTarget
class UR5GymEnv(gym.Env):
    """
    纯SAC强化学习控制的UR5机械臂Gymnasium环境。
    """
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, use_random_target=True, fixed_target=None, max_episode_steps=400, control_dt=0.05):
        super(UR5GymEnv, self).__init__()

        # ROS 2 初始化
        if not rclpy.ok():
            rclpy.init()
        self.node = Node('ur5_gym_env_node')

        # 控制周期配置
        self.control_dt = control_dt  # 控制周期（秒），默认0.05秒(20Hz)
        self._controller_dt = 0.001   # 控制器内部周期(1000Hz)
        self._spin_iterations = max(1, int(self.control_dt / self._controller_dt))  # 每次step的spin次数

        # 关节名称
        self.joint_names = [
            'shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
            'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint'
        ]

        # 动作空间：关节角度增量 (设置为 1.0，实际缩放由 Actor 处理)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32)

        # 观测空间
        obs_space_dims = 6 + 6 + 3 + 3 + 1 # q, dq, pos_ee, pos_goal, dist_p
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_space_dims,), dtype=np.float32)

        # 内部状态
        self.q_curr = np.zeros(6, dtype=np.float32)
        self.dq_curr = np.zeros(6, dtype=np.float32)
        self.pos_goal = np.zeros(3, dtype=np.float32)

        # 正运动学验证: X=0.351, Y=0.240, Z=0.481
        base_config = np.array([0.6, -1.57, 1.57, -1.57, -1.57, 0.0], dtype=np.float32)
        self.q_start = base_config.copy()

        # 单个起始构型（简化为单一基础构型，通过噪声增加多样性）
        # 起始区域要求：X > 0.2, Y > 0.2
        self.start_configs = [
            base_config,
        ]

        # 随机数生成器（必须在 _generate_new_target 之前初始化）
        self.np_random = np.random.default_rng()
        
        # 目标点配置
        self.use_random_target = use_random_target
        self.fixed_target = fixed_target
        
        # 🔴 重构：使用服务客户端替代话题订阅
        # 创建 /generate_target 服务客户端（按需同步生成模式）
        self.target_client: Client = self.node.create_client(
            GenerateTarget,
            '/generate_target'
        )
        
        # 等待服务可用（最多等待10秒）
        if not self.target_client.wait_for_service(timeout_sec=10.0):
            self.node.get_logger().warn(
                "服务 /generate_target 不可用，将使用内部目标生成作为备用"
            )
        self.node.get_logger().info("已连接到 /generate_target 服务")
        
        # 设置初始目标点 - 使用更可达的位置 [0.5, 0.0, 0.4]
        if self.fixed_target is not None:
            self.pos_goal = self.fixed_target.astype(np.float32)
        elif self.use_random_target:
            # 尝试通过服务获取初始目标点
            self._request_target_from_service()
            if np.linalg.norm(self.pos_goal) < 0.01:
                # 如果服务调用失败，使用内部生成
                self._generate_new_target()
        else:
            self.pos_goal = np.array([0.5, 0.0, 0.4], dtype=np.float32)  # 更可达的目标位置

        # 成功阈值 (从 0.2 降低到 0.05，适配精细控制)
        self.SUCCESS_THRESHOLD = 0.1

        # ROS 2 通信
        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.joint_sub = self.node.create_subscription(JointState, '/joint_states', self._joint_state_callback, qos)
        
        # 创建目标点话题订阅器（订阅 /target_position 话题）
        # 回合开始前订阅更新后的目标点，回合内保持固定
        self.target_sub = self.node.create_subscription(
            Point,
            '/target_position',
            self._target_callback,
            10
        )
        self.target_received = False  # 标记是否收到新目标点
        self.trajectory_pub = self.node.create_publisher(JointTrajectory, '/joint_trajectory_controller/joint_trajectory', 10)
        
        # 碰撞检测相关变量
        self.collision_detected = False
        self.last_collision_time = 0.0
        
        # Bumper states 订阅器 - 用于碰撞检测
        try:
            self.bumper_sub = self.node.create_subscription(
                ContactsState, 
                '/bumper_states',
                self._bumper_callback,
                10
            )
            self.node.get_logger().info("已启用 Gazebo Bumper 订阅器 (ContactsState)")
        except Exception as e:
            self.node.get_logger().warn(f"无法创建 bumper_states 订阅器: {e}")
            self.bumper_sub = None

        # TF监听器
        self.tf_buffer = Buffer(cache_time=rclpy.duration.Duration(seconds=10.0))
        self.tf_listener = TransformListener(self.tf_buffer, self.node)
        
        self.ee_frame = 'flange'
        self.base_frame = 'base_link'
        
        # 状态标志
        self.max_episode_steps = max_episode_steps
        self.current_step = 0
        self.reset_in_progress = False
        self.is_resetting = False  # 新增：用于在 reset 期间禁用检测
        
        # 动作缩放 (移除环境侧缩放，改为 1.0)
        self.prev_action = np.zeros(self.action_space.shape[0]) # 初始化为 [0, 0, 0, 0, 0, 0]
        self.action_scale = 1.0
        # self.alpha = 0.4  # 移除动作平滑系数，改用奖励约束
        
        # 添加距离跟踪
        self.prev_distance = None
        self.last_distance = None  # 用于 Episode 结束打印日志
        self.best_distance = None  # 本回合最近距离
        self.milestones = set()    # 已达成的里程碑阈值
        
        # 🔴 添加观测有效性追踪
        self._last_valid_pos_ee = np.array([0.3, 0.0, 0.5], dtype=np.float32)  # 合理默认值
        self.invalid_observation_count = 0  # 统计无效观测次数
        
        # 关节限位 (弧度) - 增加 0.01 弧度的安全余量 (Safety Margin)
        self.joint_safety_margin = 0.5 
    
        self.joint_limits = {
        # UR5 的 shoulder_lift_joint (关节2) 放宽限位，允许大臂下垂
        'shoulder_lift_joint': (-2.0, -0.5),   # 允许大臂下垂，看到更多下方目标
        
        # elbow_joint (关节3) 限制范围防止大臂小臂互撞
        'elbow_joint': (-np.pi + 0.5, np.pi - 0.5),
        
        # 其他关节保持宽松或根据实际仿真调整
        'shoulder_pan_joint': (-2 * np.pi + self.joint_safety_margin, 2 * np.pi - self.joint_safety_margin),
        'wrist_1_joint': (-2 * np.pi + self.joint_safety_margin, 2 * np.pi - self.joint_safety_margin),
        'wrist_2_joint': (-2 * np.pi + self.joint_safety_margin, 2 * np.pi - self.joint_safety_margin),
        'wrist_3_joint': (-2 * np.pi + self.joint_safety_margin, 2 * np.pi - self.joint_safety_margin)
        }
        
        # 统计计数器
        self.tolerance_violations = 0
        self.limit_violations = 0
        self.TOLERANCE_THRESHOLD = 0.2 # 控制器容差阈值
        
        # 等待 TF 树就绪 (增强版)
        self.node.get_logger().info('Waiting for TF tree to be fully ready...')
        tf_ready = False
        for i in range(10):  # 最多等待 10 秒
            if self.tf_buffer.can_transform(
                self.base_frame, 
                self.ee_frame, 
                rclpy.time.Time(), 
                timeout=rclpy.duration.Duration(seconds=1.0)
            ):
                tf_ready = True
                self.node.get_logger().info('TF tree is fully ready.')
                break
            else:
                self.node.get_logger().warn(f'TF tree not ready yet (attempt {i+1}/10), retrying...')
                rclpy.spin_once(self.node, timeout_sec=0.1)
        
        if not tf_ready:
            self.node.get_logger().error('TF tree failed to initialize within 10s. This may cause coordinate errors!')

        self.node.get_logger().info("环境初始化完成")

    def _target_callback(self, msg):
        """
        目标点话题回调函数
        
        订阅 /target_position 话题，接收新目标点坐标
        回合开始前收到新目标点后更新，回合内保持固定
        
        Args:
            msg: Point 消息，包含 x, y, z 坐标
        """
        self.pos_goal = np.array([msg.x, msg.y, msg.z], dtype=np.float32)
        self.target_received = True
        self.node.get_logger().debug(f"接收到新目标点: {self.pos_goal}")

    def _request_target_from_service(self):
        """
        通过 ROS2 服务同步请求新目标点（按需生成模式）
        
        交互流程：
        1. 发送服务请求到 /generate_target
        2. 阻塞等待响应（同步调用）
        3. 将响应中的坐标更新为当前目标点
        
        Returns:
            bool: 服务调用是否成功
        """
        if not self.target_client.service_is_ready():
            self.node.get_logger().warn("服务 /generate_target 不可用")
            return False
        
        try:
            # 创建空请求
            request = GenerateTarget.Request()
            
            # 同步调用服务（阻塞等待响应）
            self.node.get_logger().debug("调用 /generate_target 服务...")
            future = self.target_client.call_async(request)
            
            # 阻塞等待结果
            while rclpy.ok() and not future.done():
                rclpy.spin_once(self.node, timeout_sec=0.01)
            
            if future.result() is not None:
                response = future.result()
                self.pos_goal = np.array([response.x, response.y, response.z], dtype=np.float32)
                self.node.get_logger().info(f"通过服务获取目标点: {self.pos_goal}")
                return True
            else:
                self.node.get_logger().error("服务调用失败：无响应")
                return False
                
        except Exception as e:
            self.node.get_logger().error(f"服务调用异常: {e}")
            return False

    def _get_real_ee_pose_from_tf(self):
        """
        从TF系统获取真实的末端执行器位置。
        改进版：增加重试机制，设置TF有效性标志位。
        当TF查询失败时，返回缓存的上一次有效位置。
        """
        max_retries = 5
        fallback_used = False
        self._tf_valid = True  # TF有效性标志位
        
        for attempt in range(max_retries):
            try:
                if self.tf_buffer.can_transform(
                    self.base_frame,
                    self.ee_frame,
                    rclpy.time.Time(),
                    timeout=rclpy.duration.Duration(seconds=0.001)
                ):
                    transform = self.tf_buffer.lookup_transform(
                        self.base_frame,
                        self.ee_frame,
                        rclpy.time.Time()
                    )
                    
                    pos = transform.transform.translation
                    pos_ee = np.array([pos.x, pos.y, pos.z + 0.16], dtype=np.float32)
                    
                    if np.linalg.norm(pos_ee) > 0.01 and np.all(np.abs(pos_ee) < 2.0):
                        self._last_valid_pos_ee = pos_ee.copy()
                        return pos_ee
                    else:
                        fallback_used = True
                        self._tf_valid = False
                else:
                    # 主动进行额外的spin并重试
                    if attempt < max_retries - 1:
                        rclpy.spin_once(self.node, timeout_sec=0.0)
                        continue
                    fallback_used = True
                    self._tf_valid = False
                    
            except Exception as e:
                if attempt < max_retries - 1:
                    rclpy.spin_once(self.node, timeout_sec=0.0)
                    continue
                self.node.get_logger().error(f'TF lookup error after {max_retries} attempts: {e}')
                fallback_used = True
                self._tf_valid = False

        if fallback_used:
            self.invalid_observation_count += 1
            if self.invalid_observation_count % 100 == 0:
                self.node.get_logger().warn(f'累计无效TF观测次数: {self.invalid_observation_count}')
            return self._last_valid_pos_ee.copy()

    def _joint_state_callback(self, msg):
        try:
            name_to_idx = {name: i for i, name in enumerate(msg.name)}
            for i, target_name in enumerate(self.joint_names):
                if target_name in name_to_idx:
                    idx = name_to_idx[target_name]
                    self.q_curr[i] = msg.position[idx]
                    if len(msg.velocity) > idx:
                        self.dq_curr[i] = msg.velocity[idx]
        except Exception as e:
            self.node.get_logger().error(f'Joint state callback error: {e}')

    def _get_observation(self):
        pos_ee = self._get_real_ee_pose_from_tf()
        
        # 🔴 关键：验证pos_ee的有效性，防止返回[0,0,0]等无效值
        # 如果pos_ee无效，应使用缓存的上一次有效位置或合理的默认值
        if np.linalg.norm(pos_ee) < 0.01:
            self.node.get_logger().warn(f'Invalid EE position near origin: {pos_ee}, using cached value')
            pos_ee = self._last_valid_pos_ee.copy()
        
        # 验证位置是否在合理范围内
        if np.any(np.abs(pos_ee) > 2.0):
            self.node.get_logger().warn(f'EE position out of bounds: {pos_ee}, using cached value')
            pos_ee = self._last_valid_pos_ee.copy()
        
        # 🔴 使用相对距离向量（目标 - 末端）作为观测
        delta_pos = self.pos_goal - pos_ee
        dist_p = np.linalg.norm(delta_pos)
        
        # 观测空间：[q, dq, pos_ee, delta_pos, dist_p]
        # 相对距离向量比绝对坐标更容易让网络学习
        obs = np.concatenate([self.q_curr, self.dq_curr, pos_ee, delta_pos, [dist_p]])
        
        # 🔴 必须添加：验证观测维度正确性
        assert len(obs) == 19, f"Observation dimension error: expected 19, got {len(obs)}"
        
        # 验证观测值是否有限
        if not np.all(np.isfinite(obs)):
            self.node.get_logger().error(f'Non-finite observation detected: {obs}')
            obs = np.clip(obs, -1e6, 1e6)
        
        return obs.astype(np.float32)

    def _generate_new_target(self, progress=0.0):
        """
        生成随机目标点，支持课程学习。
        progress: [0, 1] 训练进度，0时最易，1时最难
        
        目标区域：左侧区域（Y负方向），与起始区域（Y正方向）分居障碍两侧
        与 target_service.py 中的 TARGET_ZONE 保持一致：
        - X: [0.4, 0.6]
        - Y: [-0.4, -0.2]
        - Z: [0.1, 0.4]
        """
        # 课程学习：随进度扩展目标区域
        x_min = 0.4
        x_max = 0.6
        y_min = -0.4 + 0.2 * progress  # 从-0.4扩展到-0.2
        y_max = -0.2
        z_min = 0.1 + 0.05 * progress  # 从0.1扩展到0.15
        z_max = 0.4
        
        max_attempts = 100
        for _ in range(max_attempts):
            x = self.np_random.uniform(x_min, x_max)
            y = self.np_random.uniform(y_min, y_max)
            z = self.np_random.uniform(z_min, z_max)
            self.pos_goal = np.array([x, y, z])
            
            if self._is_position_reachable(self.pos_goal):
                return
        
        # 如果多次尝试失败，使用默认位置
        self.node.get_logger().warn(f"Failed to generate reachable target in {max_attempts} attempts")
        self.pos_goal = np.array([-0.4, 0.0, 0.35])
    
    def _is_position_reachable(self, position):
        """
        检查笛卡尔空间位置是否可达。
        返回 True 表示可达，False 表示不可达。
        """
        # 距离基座中心检查：0.25~0.75m
        dist = np.linalg.norm(position)
        if not (0.25 < dist < 0.75):
            return False
        
        # 高度检查：不低于 0.05m，不超过 0.6m
        if position[2] < 0.05 or position[2] > 0.6:
            return False
        
        # 粗略障碍物检查：障碍物中心约在 (0, 0, 0.3)，尺寸约 0.3x0.3x0.4
        # 简单立方体障碍区检查
        obs_center = np.array([0.0, 0.0, 0.3])
        obs_half_size = np.array([0.15, 0.15, 0.2])
        if (abs(position[0] - obs_center[0]) < obs_half_size[0] and
            abs(position[1] - obs_center[1]) < obs_half_size[1] and
            abs(position[2] - obs_center[2]) < obs_half_size[2]):
            return False
        
        return True

    def reset(self, seed=None, options=None):
        # 1. 打印上一轮的距离（如果存在）
        if hasattr(self, 'last_distance') and self.last_distance is not None:
            print(f"[Episode End] End-Effector Distance to Goal: {self.last_distance:.4f} meters")
        
        super().reset(seed=seed)
        
        self.is_resetting = True  # 开始重置，禁用检测
        self.reset_in_progress = True
        self.collision_detected = False  # 重置时清零碰撞标志
        self.prev_distance = None  # 重置距离跟踪
        self.last_distance = None  # 重置标记
        self.prev_action = np.zeros(self.action_space.shape[0]) # 初始化为 [0, 0, 0, 0, 0, 0]  # 重置动作平滑缓存
        
        # 【关键修复】强制重置内存中的关节状态
        # 防止上一回合结束时的“目标位置”残留在 self.q_curr 中
        # 导致 reset 时观测到的距离异常小（一步成功）
        self.q_curr = self.q_start.copy()
        self.dq_curr = np.zeros(6, dtype=np.float32)
        
        # 重置违规统计
        self.tolerance_violations = 0
        self.limit_violations = 0
    
        # 🟢 订阅模式：每轮训练开始前请求新目标并等待订阅更新
        print("[Reset] 请求新目标点...")
        
        if self.use_random_target and not self.fixed_target:
            # 通过服务请求新目标点（触发服务端发布到 /target_position 话题）
            service_success = self._request_target_from_service()
            
            if service_success:
                # 等待订阅到更新后的目标点（最多等待3秒）
                print("[Reset] 等待订阅目标点更新...")
                self.target_received = False
                timeout = 3.0
                start_time = self.node.get_clock().now().nanoseconds / 1e9
                while not self.target_received:
                    rclpy.spin_once(self.node, timeout_sec=0.1)
                    current_time = self.node.get_clock().now().nanoseconds / 1e9
                    if current_time - start_time > timeout:
                        self.node.get_logger().warn("等待目标点超时，使用服务返回的目标点")
                        break
        
        # 如果服务调用失败或使用固定目标，使用备用方案
        if self.fixed_target is not None:
            self.pos_goal = self.fixed_target.astype(np.float32)
        elif not self.use_random_target:
            self.pos_goal = np.array([0.5, 0.0, 0.4], dtype=np.float32)
        
        print(f"[Reset] 当前目标点: {self.pos_goal}")
    
        # 从预定义起始构型候选列表中随机选择
        base_config = self.start_configs[self.np_random.integers(len(self.start_configs))]
        
        # 加上小噪声（±0.1 rad）
        noise = self.np_random.uniform(-0.1, 0.1, size=6)
        start_q = base_config + noise
        
        # 限制在安全范围内
        for i, name in enumerate(self.joint_names):
            low, high = self.joint_limits[name]
            start_q[i] = np.clip(start_q[i], low + 0.1, high - 0.1)
    
        # 移动到随机起始位置
        success = self.move_to_position(start_q)
        
        retry_count = 0
        max_retries = 2
        while not success and retry_count < max_retries:
            print(f"重置第 {retry_count + 1} 次失败，正在重试...")
            time.sleep(1.0)
            success = self.move_to_position(start_q)
            retry_count += 1
    
        if not success:
            print("警告：重置过程中机械臂未能到达初始位置，继续使用当前位置")

        # 强制进行话题同步，确保 q_curr 和 TF 缓冲区在返回前已完全更新
        # 使用仿真时间等待，确保控制器和TF更新完毕
        start_time = self.node.get_clock().now()
        target_duration = rclpy.duration.Duration(seconds=0.15)  # 等待150ms确保同步
        while (self.node.get_clock().now() - start_time) < target_duration:
            rclpy.spin_once(self.node, timeout_sec=0.0)

        # 获取最新的观测
        observation = self._get_observation()
        info = {}
        
        # 初始化距离跟踪
        self.prev_distance = observation[-1]  # 初始距离
        self.best_distance = observation[-1]
        self.milestones = set()
        
        print(f"Reset completed. Goal: {self.pos_goal}, Current EE: {observation[12:15]}, Distance: {observation[-1]}")
    
        self.current_step = 0
        self.reset_in_progress = False
        self.is_resetting = False
        self.collision_detected = False
        return observation, info

    def move_to_initial_position(self):
        """
        强制控制机器人回到初始关节位置，并等待完成。
        """
        return self.move_to_position(self.q_start)
    
    def move_to_position(self, target_q):
        """强制控制机器人移动到指定关节位置，并等待完成。"""
        print(f"Mandatory move to: {target_q}")
        current_pos = self.q_curr.copy()

        steps = 10
        for i in range(1, steps + 1):
            ratio = i / steps
            intermediate_target = current_pos + ratio * (target_q - current_pos)
            self._publish_command_with_duration(intermediate_target, duration=0.5)
            time.sleep(0.5)

        success = self.wait_for_position_reached(
            target_position=target_q,
            tolerance=0.02,
            timeout=20.0
        )
        for _ in range(5):
            rclpy.spin_once(self.node, timeout_sec=0.05)

        return success

    def _publish_command_with_duration(self, q_target, duration=0.5):
        traj_msg = JointTrajectory()
        traj_msg.joint_names = self.joint_names
        point = JointTrajectoryPoint()
        point.positions = [float(q) for q in q_target]
        point.time_from_start = Duration(
            sec=int(duration),
            nanosec=int((duration - int(duration)) * 1e9)
        )
        traj_msg.points = [point]
        self.trajectory_pub.publish(traj_msg)

    def wait_for_position_reached(self, target_position, tolerance=0.05, timeout=20.0):
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            current_pos = self.q_curr.tolist()
            
            if current_pos is not None and len(current_pos) == 6:
                position_diff = np.abs(np.array(current_pos) - np.array(target_position))
                
                if np.all(position_diff <= tolerance):
                    print(f"Arm reached initial position in: {time.time() - start_time:.2f}s")
                    return True
            
            rclpy.spin_once(self.node, timeout_sec=0.05)
            time.sleep(0.05)
            
        print(f"Warning: Arm failed to reach initial position within {timeout}s")
        print(f"Target: {target_position}")
        print(f"Current: {self.q_curr}")
        print(f"Difference: {np.abs(np.array(self.q_curr) - np.array(target_position))}")
        return False

    def step(self, action):
        self.current_step += 1
        
        # 0. 诊断与数值修正：确保关节角度在安全范围内
        for i, name in enumerate(self.joint_names):
            low, high = self.joint_limits[name]
            if self.q_curr[i] < low or self.q_curr[i] > high:
                self.limit_violations += 1
                # 强制修正（Clipping）以防止 Gazebo 报错或硬件损坏
                self.q_curr[i] = np.clip(self.q_curr[i], low, high)
                if self.limit_violations % 50 == 0: # 减少日志频率
                    self.node.get_logger().warn(f"关节 {name} 修正至: {self.q_curr[i]:.4f}")

        # 1. 动作执行 (清理冗余缩放，直接使用策略网络输出的 0.05 步长动作)
        current_action_increment = action # 已由 Actor 缩放至 [-0.05, 0.05]
        q_target = self.q_curr + current_action_increment
        
        # 增加“关节安全区”硬限制
        # 针对 shoulder_lift_joint (1) 和 elbow_joint (2)，与 joint_limits 一致
        q_target[1] = np.max([q_target[1], -2.0])   # 与 joint_limits 一致
        q_target[2] = np.max([q_target[2], -2.4])

        # 确保目标角度不超限
        for i, name in enumerate(self.joint_names):
            low, high = self.joint_limits[name]
            q_target[i] = np.clip(q_target[i], low, high)
            
        self._publish_command(q_target)
        
        # 2. 严格仿真时间等待 - 确保经过完整的control_dt时长
        start_time = self.node.get_clock().now()
        target_duration = rclpy.duration.Duration(seconds=self.control_dt)
        while (self.node.get_clock().now() - start_time) < target_duration:
            rclpy.spin_once(self.node, timeout_sec=0.0)

        # 3. 获取新的观测与容差检查
        new_obs = self._get_observation()
        
        # 🔴 安全地解析观测各分量
        q_curr_new = new_obs[0:6]
        dq_curr_new = new_obs[6:12]  
        new_pos_ee = new_obs[12:15]  # 末端位置
        delta_pos = new_obs[15:18]   # 相对距离向量（目标 - 末端）
        new_dist = float(new_obs[18]) # 距离
        
        # 从相对距离向量计算绝对目标位置
        pos_goal = new_pos_ee + delta_pos
        
        # 🔴 添加观测合理性检查
        # 如果末端位置明显不合理（如接近原点但目标很远），给予惩罚或特殊处理
        invalid_observation_penalty = 0.0
        if np.linalg.norm(new_pos_ee) < 0.1 and np.linalg.norm(pos_goal) > 0.3:
            # 这可能表示TF获取失败，记录警告并施加惩罚
            self.node.get_logger().warn(f'Suspicious observation: EE near origin ({new_pos_ee}) but goal far ({pos_goal})')
            invalid_observation_penalty = -10.0
        
        # 计算位置误差
        joint_errors = np.abs(self.q_curr - q_target)
        if np.any(joint_errors > self.TOLERANCE_THRESHOLD):
            self.tolerance_violations += 1
        
        # 存储当前距离供 Episode 结束时打印
        self.last_distance = new_dist

        # 4. 检查终止条件
        terminated = False
        truncated = False
        reward = 0.0
        info = {
            'distance': new_dist,
            'collision': False,
            'success': False,
            'timeout': False,
            'steps': self.current_step,
            'reason': 'in_progress',
            'q_curr': self.q_curr.copy(),
            'dq_curr': self.dq_curr.copy(),
            'q_target': q_target,
            'tolerance_violations': self.tolerance_violations,
            'limit_violations': self.limit_violations,
            'joint_errors': joint_errors
        }

        # 4.1 检查碰撞
        if self.collision_detected:
            reward = -10.0
            terminated = True
            info['collision'] = True
            info['reason'] = 'collision'
            self.prev_action = action.copy()
            return new_obs, reward, terminated, truncated, info

        # 4.2 检查成功
        if new_dist <= self.SUCCESS_THRESHOLD:
            reward = 200.0
            terminated = True
            info['success'] = True
            info['reason'] = 'success'
            self.prev_action = action.copy()
            return new_obs, reward, terminated, truncated, info

        # 4.3 检查超时 → 固定惩罚，让 shaping 信号主导学习
        if self.max_episode_steps and self.current_step >= self.max_episode_steps:
            reward = -3.0
            terminated = True
            info['timeout'] = True
            info['reason'] = 'timeout'
            self.prev_action = action.copy()
            return new_obs, reward, terminated, truncated, info

        # 5. 势能差分奖励 (potential-based shaping)
        if self.prev_distance is not None:
            potential_old = -self.prev_distance
            potential_new = -new_dist
            reward = (potential_new - potential_old) * 30.0
        else:
            reward = 0.0

        # 5.1 基座过近惩罚 (只管危险范围，不管外扩激励)
        ee_base_dist = np.linalg.norm(new_pos_ee)
        if ee_base_dist < 0.35:
            reward -= (0.35 - ee_base_dist) * 15.0

        # 5.2 基座过近直接终止
        if ee_base_dist < 0.15:
            reward = -80.0
            terminated = True
            info['reason'] = 'base_proximity'
            self.prev_action = action.copy()
            return new_obs, reward, terminated, truncated, info

        # 5.3 里程碑奖励：首次突破距离阈值时给正反馈
        MILESTONE_THRESHOLDS = [0.7, 0.5, 0.35, 0.2]
        MILESTONE_BONUS = 5.0
        if new_dist < self.best_distance:
            self.best_distance = new_dist
            for thresh in MILESTONE_THRESHOLDS:
                if new_dist <= thresh and thresh not in self.milestones:
                    self.milestones.add(thresh)
                    reward += MILESTONE_BONUS

        reward += invalid_observation_penalty

        # 6. 状态更新
        self.prev_distance = new_dist
        self.prev_action = action.copy()

        return new_obs, reward, terminated, truncated, info

    def _publish_command(self, q_target):
        self._publish_command_with_duration(q_target, duration=0.1)

    # --- 替换开始 ---
    def _bumper_callback(self, msg):
        """
        处理 Gazebo ContactsState 消息。
        过滤自碰撞（UR5 连杆之间的接触），仅检测与环境物体的碰撞。
        """
        if self.is_resetting:
            return

        try:
            uri_prefixes = ['ur5', 'ur_', 'robot', 'base_link', 'shoulder',
                          'upper_arm', 'forearm', 'wrist', 'flange', 'tool0']
            
            real_collision = False
            for state in msg.states:
                c1 = state.collision1_name.lower()
                c2 = state.collision2_name.lower()
                
                c1_is_ur5 = any(p in c1 for p in uri_prefixes)
                c2_is_ur5 = any(p in c2 for p in uri_prefixes)
                
                if c1_is_ur5 and c2_is_ur5:
                    continue
                
                real_collision = True
                self.node.get_logger().warn(
                    f'碰撞接触: {state.collision1_name} <-> {state.collision2_name}'
                    f' (力: {state.total_wrench.force.z:.2f}N)'
                )
                break

            if real_collision:
                current_time = time.time()
                if current_time - self.last_collision_time > 0.1:
                    self.node.get_logger().warn('真实碰撞检测触发！')
                    self.collision_detected = True
                    self.last_collision_time = current_time

        except Exception as e:
            self.node.get_logger().error(f'Bumper callback error: {e}')
    # --- 替换结束 ---

    def close(self):
        if self.node:
            self.node.destroy_node()

    def debug_info(self):
        """调试函数：打印当前状态信息"""
        obs = self._get_observation()
        pos_ee = obs[12:15]
        pos_goal = obs[15:18]
        dist = obs[-1]
        print(f"EE Position: {pos_ee}")
        print(f"Goal Position: {pos_goal}")
        print(f"Distance: {dist}")
        print(f"Joint Angles: {obs[:6]}")
        print(f"Collision Detected: {self.collision_detected}")
        print(f"Previous Distance: {self.prev_distance}")
