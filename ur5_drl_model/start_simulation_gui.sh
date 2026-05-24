#!/bin/bash
# 启动 UR5 仿真（带 GUI）

# 创建临时工作目录
SIM_DIR="/tmp/ur5_sim_gui_$(date +%s)"
mkdir -p "$SIM_DIR/logs" "$SIM_DIR/ros_logs" "$SIM_DIR/.config" "$SIM_DIR/.cache"

echo "=== Starting UR5 simulation with GUI ==="
echo "Temp directory: $SIM_DIR"

# 设置环境变量
export HOME="$SIM_DIR"
export ROS_LOG_DIR="$SIM_DIR/ros_logs"
export GAZEBO_LOG_DIR="$SIM_DIR/logs"
export GAZEBO_MODEL_PATH="/usr/share/gazebo-11/models:/opt/ros/humble/share"

# GUI 相关环境变量
export DISPLAY="${DISPLAY:-:0}"
export XAUTHORITY="${XAUTHORITY:-$HOME/.Xauthority}"
export QT_QPA_PLATFORM="xcb"
export QT_FONT_DPI=96
export DCONF_USER_CONFIG_DIR="$SIM_DIR/.config"
export DCONF_CACHE_DIR="$SIM_DIR/.cache"

# 创建 dconf 所需的目录和文件
mkdir -p "$SIM_DIR/.config/dconf"
mkdir -p "$SIM_DIR/.cache/dconf"
touch "$SIM_DIR/.config/dconf/user"

# 进入项目目录
cd /home/gwh/ros2_projects/ur5_drl_model

# 加载 ROS2 环境
source /opt/ros/humble/setup.bash

# 加载项目环境
source install/setup.bash

# 打印环境变量验证
echo "DISPLAY=$DISPLAY"
echo "HOME=$HOME"
echo "ROS_LOG_DIR=$ROS_LOG_DIR"
echo "GAZEBO_LOG_DIR=$GAZEBO_LOG_DIR"

# 启动仿真（带 GUI）
echo "Starting simulation with GUI..."
ros2 launch ur_simulation_gazebo ur_sim_control.launch.py ur_type:=ur5 gazebo_gui:=true

# 清理临时目录
echo "Cleaning up..."
rm -rf "$SIM_DIR"
