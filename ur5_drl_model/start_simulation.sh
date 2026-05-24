#!/bin/bash
# 启动 UR5 仿真（带 GUI 和 RViz）

# 创建临时工作目录
SIM_DIR="/tmp/ur5_sim_$(date +%s)"
mkdir -p "$SIM_DIR/logs" "$SIM_DIR/ros_logs" "$SIM_DIR/.config/dconf" "$SIM_DIR/.cache/dconf"

echo "=== Starting UR5 simulation ==="
echo "Temp directory: $SIM_DIR"

# 设置环境变量
export HOME="$SIM_DIR"
export ROS_LOG_DIR="$SIM_DIR/ros_logs"
export GAZEBO_LOG_DIR="$SIM_DIR/logs"
export GAZEBO_MODEL_PATH="/usr/share/gazebo-11/models:/opt/ros/humble/share"

# GUI 相关环境变量
export DISPLAY="${DISPLAY:-:0}"
export XAUTHORITY="${XAUTHORITY:-$HOME/.Xauthority}"
export DCONF_USER_CONFIG_DIR="$SIM_DIR/.config"
export DCONF_CACHE_DIR="$SIM_DIR/.cache"
touch "$SIM_DIR/.config/dconf/user"

# 进入项目目录
cd /home/gwh/SAC/ur5_drl_model

# 加载 ROS2 环境
source /opt/ros/humble/setup.bash

# 加载项目环境
source install/setup.bash

# 启动仿真（默认带 GUI 和 RViz）
echo "Starting simulation with GUI and RViz..."
ros2 launch ur_simulation_gazebo ur_sim_control.launch.py ur_type:=ur5

# 清理临时目录
echo "Cleaning up..."
rm -rf "$SIM_DIR"