#!/bin/bash

# UR5 DRL 训练一键启动脚本
# 使用方法: bash run_training.sh
# 功能: 自动启动仿真环境、目标点服务和模型训练
# 注意: 仿真和目标点服务在系统环境下运行，SAC训练在虚拟环境下运行

echo "======================================"
echo "  UR5 DRL 训练一键启动脚本"
echo "======================================"

# 创建临时目录用于日志和配置
mkdir -p /tmp/ur5_drl_logs/ros_logs
mkdir -p /tmp/ur5_drl_logs/gazebo_logs
mkdir -p /tmp/ur5_sim_home/.config/dconf
mkdir -p /tmp/ur5_sim_home/.cache/dconf
mkdir -p /tmp/ur5_sim_home/.Xauthority

# 关键环境变量设置（解决容器环境中的显示和配置问题）
export HOME=/tmp/ur5_sim_home
export XAUTHORITY=/run/user/1000/.mutter-Xwaylandauth.IYUOP3
export DISPLAY=:0
export DCONF_USER_CONFIG_DIR=/tmp/ur5_sim_home/.config/dconf
export DCONF_CACHE_DIR=/tmp/ur5_sim_home/.cache/dconf
export ROS_LOG_DIR=/tmp/ur5_drl_logs/ros_logs
export GAZEBO_LOG_DIR=/tmp/ur5_drl_logs/gazebo_logs

# 尝试禁用 wayland 强制使用 X11（解决某些环境中的显示问题）
export XDG_SESSION_TYPE=x11
export QT_QPA_PLATFORM=xcb

echo "环境配置完成"
echo "HOME: $HOME"
echo "DISPLAY: $DISPLAY"
echo "XAUTHORITY: $XAUTHORITY"

# 激活 ROS2 环境（系统环境）
source /opt/ros/humble/setup.bash

# 激活仿真模型包
source /home/gwh/SAC/ur5_drl_model/install/setup.bash

# 激活训练环境包
source /home/gwh/SAC/ur5_drl_agent/install/setup.bash

echo ""
echo "[步骤1/3] 启动 Gazebo 仿真环境..."
echo "⚠️  Gazebo 和 RViz 窗口应该会显示在屏幕上"

# 启动仿真（带 GUI 和 RViz）- 在系统环境下运行
ros2 launch ur_simulation_gazebo ur_sim_control.launch.py ur_type:=ur5 gazebo_gui:=true &
SIM_PID=$!
echo "仿真进程 PID: $SIM_PID"

# 等待仿真启动（25秒，确保 Gazebo、RViz 和 controller_manager 完全就绪）
echo "等待仿真环境初始化（25秒）..."
sleep 25

echo ""
echo "[步骤2/3] 启动目标点生成服务..."
# 使用系统 Python 运行目标点服务（避免使用虚拟环境）
/usr/bin/python3 /home/gwh/SAC/ur5_drl_model/install/ur_simulation_gazebo/lib/ur_simulation_gazebo/target_service.py &
TARGET_PID=$!
echo "目标点服务进程 PID: $TARGET_PID"

# 等待服务启动（5秒）
echo "等待目标点服务启动..."
sleep 5

# 检查服务是否可用
echo "检查服务状态..."
ros2 service list | grep generate_target && echo "✓ 服务已启动" || echo "✗ 服务启动失败"

echo ""
echo "[步骤3/3] 启动 SAC 训练..."
# 使用子shell运行训练，确保虚拟环境不会污染父进程
(
    # 在子shell中激活虚拟环境
    source /home/gwh/SAC/ur5_drl_agent/venv/bin/activate
    
    # 设置 Python 路径
    export PYTHONPATH=/home/gwh/SAC/ur5_drl_agent/install/lib/python3.10/site-packages:$PYTHONPATH
    
    # 运行训练
    python3 -m ur5_drl_env.train_sac
)
TRAIN_EXIT_CODE=$?

echo ""
echo "======================================"
echo "  训练结束，清理后台进程..."
echo "======================================"

# 终止后台进程
kill $TARGET_PID 2>/dev/null
kill $SIM_PID 2>/dev/null

echo "完成！"