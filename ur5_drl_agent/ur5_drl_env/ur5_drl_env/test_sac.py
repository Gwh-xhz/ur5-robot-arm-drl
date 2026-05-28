import rclpy
import numpy as np
import time
import sys
import os

if "/tmp/pip_packages" not in sys.path:
    sys.path.append("/tmp/pip_packages")

from stable_baselines3 import SAC
from .ur5_gym_env import UR5GymEnv
from .train_sac import CustomSACPolicy

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
from rich.layout import Layout

console = Console()

TOTAL_TEST_EPISODES = 100
MODEL_DIR = "/home/gwh/SAC/models/sac_ur5"
MODEL_PATH = ""
for _f in os.listdir(MODEL_DIR) if os.path.isdir(MODEL_DIR) else []:
    if _f.endswith(".zip") and "sac_ur5_final" in _f:
        MODEL_PATH = os.path.join(MODEL_DIR, _f)
        break


def run_test():
    if not rclpy.ok():
        rclpy.init()

    env = UR5GymEnv(use_random_target=True, max_episode_steps=400, control_dt=0.05)

    console.print("[yellow]等待环境初始化 (TF 树和控制器状态)...[/yellow]")
    for _ in range(50):
        rclpy.spin_once(env.node, timeout_sec=0.1)
    time.sleep(2.0)

    console.print("[cyan]检查TF变换可用性...[/cyan]")
    tf_available = False
    for _ in range(30):
        try:
            env.tf_buffer.lookup_transform(
                'base_link', 'flange',
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.1)
            )
            tf_available = True
            console.print("[green]TF变换可用[/green]")
            break
        except Exception:
            time.sleep(0.1)

    if not tf_available:
        console.print("[red]TF变换不可用，测试终止！[/red]")
        env.close()
        if rclpy.ok():
            rclpy.shutdown()
        return

    console.print(f"[cyan]模型路径: {MODEL_PATH}[/cyan]")
    if not os.path.exists(MODEL_PATH):
        console.print(f"[red]模型文件不存在: {MODEL_PATH}[/red]")
        env.close()
        if rclpy.ok():
            rclpy.shutdown()
        return

    try:
        model = SAC.load(MODEL_PATH, env=env, device='cpu',
                         custom_objects={'policy_class': CustomSACPolicy})
        console.print(Panel(f"成功加载模型", title="Model Loading", border_style="green"))
    except Exception as e:
        console.print(Panel(f"加载模型失败: {e}", title="Error", border_style="red"))
        env.close()
        if rclpy.ok():
            rclpy.shutdown()
        return

    stats = {
        'success': 0,
        'collision': 0,
        'timeout': 0,
        'base_proximity': 0,
        'joint_limit': 0,
        'other': 0,
        'rewards': [],
        'steps': [],
        'final_distances': [],
        'initial_distances': [],
        'goal_positions': [],
    }

    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=None),
        TaskProgressColumn(),
        MofNCompleteColumn(),
        TimeRemainingColumn(),
        expand=True
    )
    task = progress.add_task("[green]Testing SAC (No-Obstacle)", total=TOTAL_TEST_EPISODES)

    def create_summary_table(ep, succ, coll, tout, other_cnt, succ_rate,
                              cur_reward, avg_reward, cur_steps, cur_dist, reason):
        table = Table(title=f"UR5 SAC Test  Episode {ep+1}/{TOTAL_TEST_EPISODES}",
                      show_header=True, header_style="bold magenta")
        table.add_column("Metric", style="dim", width=18)
        table.add_column("Value", width=25)

        reason_color = {"success": "green", "collision": "red",
                         "timeout": "yellow", "base_proximity": "red",
                         "joint_limit": "yellow"}.get(reason, "white")

        table.add_row("Success Count", f"[green]{succ}[/green]")
        table.add_row("Collision Count", f"[red]{coll}[/red]")
        table.add_row("Timeout Count", f"[yellow]{tout + other_cnt}[/yellow]")
        table.add_row("Success Rate", f"[bold]{succ_rate:.1f}%[/bold]")
        table.add_row("Episode Steps", f"{cur_steps}")
        table.add_row("Episode Reward", f"{cur_reward:.2f}")
        table.add_row("Avg Reward (all)", f"{avg_reward:.2f}")
        table.add_row("Final Distance", f"{cur_dist:.4f} m")
        table.add_row("Result", f"[{reason_color}]{reason}[/{reason_color}]")

        return table

    try:
        with Live(console=console, refresh_per_second=4) as live:
            for ep in range(TOTAL_TEST_EPISODES):
                obs, _ = env.reset()
                initial_dist = obs[-1]
                done = False
                truncated = False
                ep_steps = 0
                ep_reward = 0.0
                reason = 'in_progress'

                while not (done or truncated):
                    rclpy.spin_once(env.node, timeout_sec=0.01)
                    action, _ = model.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, info = env.step(action)
                    ep_reward += reward
                    ep_steps += 1
                    done = terminated or truncated

                reason = info.get('reason', 'timeout')
                final_dist = info.get('distance', obs[-1])

                if info.get('success', False):
                    stats['success'] += 1
                elif info.get('collision', False):
                    stats['collision'] += 1
                elif reason == 'timeout':
                    stats['timeout'] += 1
                elif reason == 'base_proximity':
                    stats['base_proximity'] += 1
                elif reason == 'joint_limit':
                    stats['joint_limit'] += 1
                else:
                    stats['other'] += 1

                stats['rewards'].append(ep_reward)
                stats['steps'].append(ep_steps)
                stats['final_distances'].append(final_dist)
                stats['initial_distances'].append(initial_dist)
                stats['goal_positions'].append(obs[15:18].copy())

                succ_rate = (stats['success'] / (ep + 1)) * 100
                avg_rew = np.mean(stats['rewards'])

                other_cnt = stats['base_proximity'] + stats['joint_limit'] + stats['other']

                layout = Layout()
                layout.split_column(
                    Layout(progress, name="progress", size=3),
                    Layout(create_summary_table(
                        ep, stats['success'], stats['collision'],
                        stats['timeout'], other_cnt, succ_rate,
                        ep_reward, avg_rew, ep_steps, final_dist, reason
                    ), name="info")
                )
                live.update(layout)
                progress.update(task, advance=1)

                res_color = {"success": "green", "collision": "red",
                              "timeout": "yellow", "base_proximity": "red",
                              "joint_limit": "yellow"}.get(reason, "white")
                console.log(
                    f"[blue]Ep {ep+1:3d}/{TOTAL_TEST_EPISODES}[/blue] | "
                    f"[{res_color}]{reason:14s}[/{res_color}] | "
                    f"[cyan]InitDist: {initial_dist:.4f}m[/cyan] | "
                    f"[cyan]FinalDist: {final_dist:.4f}m[/cyan] | "
                    f"[white]Steps: {ep_steps:3d}[/white] | "
                    f"[magenta]Reward: {ep_reward:7.2f}[/magenta] | "
                    f"[yellow]SR: {succ_rate:.1f}%[/yellow]"
                )

    except KeyboardInterrupt:
        console.print("\n[red]测试被用户中断[/red]")

    # ==========================================
    # 最终汇总
    # ==========================================
    total = len(stats['rewards'])
    if total == 0:
        console.print("[red]无有效测试数据[/red]")
        env.close()
        if rclpy.ok():
            rclpy.shutdown()
        return

    succ = stats['success']
    coll = stats['collision']
    tout = stats['timeout']
    base_p = stats['base_proximity']
    jlim = stats['joint_limit']
    oth = stats['other']
    rewards_arr = np.array(stats['rewards'])
    steps_arr = np.array(stats['steps'])
    dists_arr = np.array(stats['final_distances'])
    init_dists_arr = np.array(stats['initial_distances'])

    summary = Table(title="UR5 SAC 100-Episode Test Summary (No-Obstacle Model)",
                    show_header=True, header_style="bold cyan")
    summary.add_column("Metric", style="dim", width=22)
    summary.add_column("Value", width=22)

    summary.add_row("Total Episodes", f"{total}")
    summary.add_row("Success", f"[green]{succ} ({succ/total*100:.1f}%)[/green]")
    summary.add_row("Collision", f"[red]{coll} ({coll/total*100:.1f}%)[/red]")
    summary.add_row("Timeout", f"[yellow]{tout} ({tout/total*100:.1f}%)[/yellow]")
    summary.add_row("Base Proximity", f"[red]{base_p}[/red]")
    summary.add_row("Joint Limit", f"[yellow]{jlim}[/yellow]")
    summary.add_row("Other", f"{oth}")
    summary.add_row("Success Rate", f"[bold green]{succ/total*100:.1f}%[/bold green]")
    summary.add_row("", "")

    summary.add_row("Avg Reward", f"{rewards_arr.mean():.2f}")
    summary.add_row("Std Reward", f"{rewards_arr.std():.2f}")
    summary.add_row("Min Reward", f"{rewards_arr.min():.2f}")
    summary.add_row("Max Reward", f"{rewards_arr.max():.2f}")
    summary.add_row("", "")

    summary.add_row("Avg Steps", f"{steps_arr.mean():.1f}")
    summary.add_row("Min Steps", f"{steps_arr.min()}")
    summary.add_row("Max Steps", f"{steps_arr.max()}")
    summary.add_row("", "")

    summary.add_row("Avg Init Distance", f"{init_dists_arr.mean():.4f} m")
    summary.add_row("Avg Final Distance", f"{dists_arr.mean():.4f} m")
    summary.add_row("Min Final Distance", f"{dists_arr.min():.4f} m")
    summary.add_row("Max Final Distance", f"{dists_arr.max():.4f} m")

    console.print(summary)

    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), "logs")
    os.makedirs(output_dir, exist_ok=True)
    summary_file = os.path.join(output_dir, f"test_result_{int(time.time())}.txt")
    with open(summary_file, 'w') as f:
        f.write(f"UR5 SAC 100-Episode Test Result\n")
        f.write(f"Model: no-obstacle (sac_ur5_final)\n")
        f.write(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"{'='*50}\n")
        f.write(f"Success Rate: {succ/total*100:.1f}% ({succ}/{total})\n")
        f.write(f"Collision:    {coll} ({coll/total*100:.1f}%)\n")
        f.write(f"Timeout:      {tout} ({tout/total*100:.1f}%)\n")
        f.write(f"Avg Reward:   {rewards_arr.mean():.2f} +/- {rewards_arr.std():.2f}\n")
        f.write(f"Avg Steps:    {steps_arr.mean():.1f}\n")
        f.write(f"Avg Distance: {dists_arr.mean():.4f} m\n")

    console.print(f"[green]结果已保存至: {summary_file}[/green]")

    env.close()
    if rclpy.ok():
        rclpy.shutdown()
    console.print("[green]资源已释放[/green]")


if __name__ == '__main__':
    run_test()