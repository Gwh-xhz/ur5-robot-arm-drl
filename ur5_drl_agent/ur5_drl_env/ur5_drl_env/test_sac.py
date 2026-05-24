import rclpy
import numpy as np
import pandas as pd
import time
import sys
import os

# 添加临时安装的 rich 库路径
if "/tmp/pip_packages" not in sys.path:
    sys.path.append("/tmp/pip_packages")

from stable_baselines3 import SAC
from .ur5_gym_env import UR5GymEnv 

# 导入 rich 进度条相关组件
from rich.progress import (
    Progress,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeRemainingColumn,
    SpinnerColumn,
    MofNCompleteColumn
)
from rich.console import Console
from rich.panel import Panel
from rich.live import Live
from rich.table import Table

console = Console()

def run_test():
    # ==========================================
    # 1. 初始化 ROS 2
    # ==========================================
    if not rclpy.ok():
        rclpy.init()
    
    # ==========================================
    # 2. 设置固定目标点
    # ==========================================
    fixed_target_point = np.array([0.5, 0.0, 0.4], dtype=np.float32)
    
    # ==========================================
    # 3. 创建环境
    # ==========================================
    env = UR5GymEnv(use_random_target=False, fixed_target=fixed_target_point)
    
    # 关键修改：大幅增加等待时间
    console.print("[yellow]⏳ 等待环境初始化 (TF 树和控制器状态)...[/yellow]")
    for _ in range(50):
        rclpy.spin_once(env.node, timeout_sec=0.1)
    time.sleep(2.0)
    
    # 额外检查TF变换是否可用
    console.print("[cyan]🔍 检查TF变换可用性...[/cyan]")
    tf_available = False
    for _ in range(30):
        try:
            transform = env.tf_buffer.lookup_transform(
                'base_link', 'flange', 
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.1)
            )
            tf_available = True
            console.print("[green]✅ TF变换可用[/green]")
            break
        except:
            time.sleep(0.1)
    
    if not tf_available:
        console.print("[red]❌ TF变换不可用，这可能导致测试失败！[/red]")
        env.close()
        if rclpy.ok():
            rclpy.shutdown()
        return

    # ==========================================
    # 4. 加载模型并验证环境
    # ==========================================
    model_path = "./models/sac_ur5/sac_ur5_final.zip"
    try:
        model = SAC.load(model_path, env=env, device='cpu')
        console.print(Panel(f"成功加载模型: {model_path}", title="Model Loading", border_style="green"))
    except Exception as e:
        console.print(Panel(f"加载模型失败: {e}", title="Model Loading Error", border_style="red"))
        return

    # ==========================================
    # 5. 初始化轨迹记录容器
    # ==========================================
    log_data = {
        'time_step': [],
        'q1': [], 'q2': [], 'q3': [], 'q4': [], 'q5': [], 'q6': [],
        'dq1': [], 'dq2': [], 'dq3': [], 'dq4': [], 'dq5': [], 'dq6': [],
        'ee_x': [], 'ee_y': [], 'ee_z': [],
        'goal_x': [], 'goal_y': [], 'goal_z': [],
        'distance_to_target': []
    }
    
    episodes = 100
    
    # 定义 Rich 进度条
    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=None),
        TaskProgressColumn(),
        MofNCompleteColumn(),
        TimeRemainingColumn(),
        expand=True
    )
    task = progress.add_task("[green]Testing SAC", total=episodes)

    # 定义监控信息表格
    def create_info_table(ep, ep_reward, ep_steps, dist, reason):
        table = Table(title=f"UR5 SAC Test Episode {ep+1}", show_header=True, header_style="bold magenta")
        table.add_column("Metric", style="dim", width=15)
        table.add_column("Value", width=25)
        
        reason_color = "white"
        if reason == 'success': reason_color = "green"
        elif reason == 'collision': reason_color = "red"
        elif reason == 'timeout': reason_color = "yellow"

        table.add_row("Step Counter", f"{ep_steps}")
        table.add_row("Reward", f"{ep_reward:.2f}")
        table.add_row("Current Dist", f"{dist:.4f} m")
        table.add_row("Result", f"[{reason_color}]{reason}[/{reason_color}]")
        
        return table

    try:
        with Live(console=console, refresh_per_second=4) as live:
            for ep in range(episodes):
                obs, _ = env.reset()
                done = False
                truncated = False
                step_counter = 0 
                episode_reward = 0

                while not (done or truncated):
                    rclpy.spin_once(env.node, timeout_sec=0.01)

                    action, _states = model.predict(obs, deterministic=True)
                    obs, reward, done, truncated, info = env.step(action)
                    episode_reward += reward

                    # 记录数据
                    q_angles = obs[:6]
                    q_velocities = obs[6:12]
                    ee_pos = obs[12:15]
                    goal_pos = obs[15:18]
                    dist = obs[-1]

                    log_data['time_step'].append(step_counter)
                    log_data['q1'].append(q_angles[0])
                    log_data['q2'].append(q_angles[1])
                    log_data['q3'].append(q_angles[2])
                    log_data['q4'].append(q_angles[3])
                    log_data['q5'].append(q_angles[4])
                    log_data['q6'].append(q_angles[5])
                    
                    log_data['dq1'].append(q_velocities[0])
                    log_data['dq2'].append(q_velocities[1])
                    log_data['dq3'].append(q_velocities[2])
                    log_data['dq4'].append(q_velocities[3])
                    log_data['dq5'].append(q_velocities[4])
                    log_data['dq6'].append(q_velocities[5])
                    
                    log_data['ee_x'].append(ee_pos[0])
                    log_data['ee_y'].append(ee_pos[1])
                    log_data['ee_z'].append(ee_pos[2])
                    
                    log_data['goal_x'].append(goal_pos[0])
                    log_data['goal_y'].append(goal_pos[1])
                    log_data['goal_z'].append(goal_pos[2])
                    
                    log_data['distance_to_target'].append(dist)

                    step_counter += 1
                    
                    # 更新 Live 显示内容
                    reason = info.get('reason', 'in_progress')
                    from rich.layout import Layout
                    layout = Layout()
                    layout.split_column(
                        Layout(progress, name="progress", size=3),
                        Layout(create_info_table(ep, episode_reward, step_counter, dist, reason), name="info")
                    )
                    live.update(layout)

                progress.update(task, advance=1)

    except KeyboardInterrupt:
        console.print("\n[red]🛑 测试被用户中断[/red]")
    finally:
        # ==========================================
        # 7. 保存数据到 CSV
        # ==========================================
        console.print("\n[yellow]💾 正在保存轨迹数据...[/yellow]")
        df = pd.DataFrame(log_data)
        filename = f"ur5_trajectory_log_{int(time.time())}.csv"
        df.to_csv(filename, index=False)
        console.print(f"[green]✅ 数据已保存至: {filename}[/green]")

        # 清理
        env.close()
        if rclpy.ok():
            rclpy.shutdown()
        console.print("[green]✅ 资源已释放[/green]")

if __name__ == '__main__':
    run_test()