import os
import sys
# 添加临时安装的 rich 库路径
if "/tmp/pip_packages" not in sys.path:
    sys.path.append("/tmp/pip_packages")

import rclpy
import numpy as np
import torch
from colorama import init, Fore, Style
from typing import Optional
import torch.nn as nn
from stable_baselines3 import SAC
from stable_baselines3.common.logger import configure
from stable_baselines3.sac.policies import MlpPolicy, Actor
from .ur5_gym_env import UR5GymEnv

# y = tanh(u) * s; log p(y) = log p(tanh(u)) - n_dims * log(s)
class CustomActor(Actor):
    ACTION_LIMIT = 0.05

    def forward(self, obs: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        action = super().forward(obs, deterministic=deterministic)
        return action * self.ACTION_LIMIT

    def action_log_prob(self, obs: torch.Tensor) -> tuple:
        tanh_action, log_prob = super().action_log_prob(obs)
        scaled_action = tanh_action * self.ACTION_LIMIT
        n_dims = tanh_action.shape[-1]
        log_prob = log_prob - torch.log(torch.tensor(self.ACTION_LIMIT, device=log_prob.device)) * n_dims
        return scaled_action, log_prob

# 自定义策略类，使用上面定义的 CustomActor
class CustomSACPolicy(MlpPolicy):
    def make_actor(self, features_extractor: Optional[nn.Module] = None) -> CustomActor:
        # 直接调用父类方法创建基础 Actor
        # MlpPolicy.make_actor 会处理 features_extractor 和所有初始化参数
        base_actor = super().make_actor(features_extractor)
        
        # 将基础 Actor 的参数和结构“转移”给 CustomActor
        # 这样可以完美避开初始化顺序导致的属性缺失问题
        custom_actor = CustomActor(
            base_actor.observation_space,
            base_actor.action_space,
            base_actor.net_arch,
            base_actor.features_extractor,
            base_actor.features_dim,
            base_actor.activation_fn,
            base_actor.use_sde,
            base_actor.log_std_init,
            base_actor.full_std,
            base_actor.use_expln,
            base_actor.clip_mean,
        )
        # 同步状态字典（权重）
        custom_actor.load_state_dict(base_actor.state_dict())
        return custom_actor

# 导入 rich 组件
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    Progress,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeRemainingColumn,
    SpinnerColumn,
    MofNCompleteColumn
)
from rich.live import Live
from rich.table import Table

# 初始化 colorama
init(autoreset=True)
console = Console()

def main():
    # 1. 初始化 ROS 2
    if not rclpy.ok():
        rclpy.init()

    # 2. 定义路径
    log_dir = "./logs/sac_ur5_tensorboard/"
    model_dir = "./models/sac_ur5/"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    # 3. 创建环境
    # 设置 max_episode_steps 为 200，control_dt 为 0.05秒(20Hz)，与控制器1000Hz解耦
    env = UR5GymEnv(use_random_target=True, max_episode_steps=400, control_dt=0.05)

    # 收集阶段动作缩放系数 (比学习阶段大，增大探索空间)
    COLLECTION_ACTION_LIMIT = 0.12
    
    # 4. 初始化 SAC 模型
    learning_starts = 10000
    batch_size = 512
    train_freq = 16
    gradient_steps = 8
    
    model = SAC(
        CustomSACPolicy,
        env,
        verbose=0,
        tensorboard_log=log_dir,
        learning_rate=1e-4,
        buffer_size=200000,
        batch_size=batch_size,
        ent_coef=0.0001,
        train_freq=train_freq,
        gradient_steps=gradient_steps,
        tau=0.005,
        gamma=0.99,
        learning_starts=learning_starts,
        policy_kwargs=dict(net_arch=[256, 256, 256])
    )

    # 手动配置 Logger
    new_logger = configure(log_dir, ["stdout", "csv", "tensorboard"])
    model.set_logger(new_logger)

    # 环境自检：用随机策略测试环境
    console.print(Panel("开始环境自检（随机策略测试）...", title="Self-Check", border_style="yellow"))
    test_distances = []
    for test_ep in range(5):
        test_obs, _ = env.reset()
        for test_step in range(200):
            test_action = env.action_space.sample() * CustomActor.ACTION_LIMIT
            test_next_obs, _, test_done, _, test_info = env.step(test_action)
            if test_done:
                break
        test_distances.append(test_info.get('distance', 999.0))
    
    avg_test_dist = np.mean(test_distances)
    console.print(f"[yellow]随机策略平均终距: {avg_test_dist:.4f}m[/yellow]")
    if avg_test_dist > 0.5:
        console.print("[red]警告: 随机策略平均终距 > 0.5m，请检查TF/目标生成逻辑![/red]")
    else:
        console.print("[green]环境自检通过[/green]")

    # 5. 训练循环统计量
    total_episodes = 5000
    success_count = 0
    collision_count = 0
    total_steps = 0
    episode_rewards = []
    action_std_history = []
    final_distances = []

    consecutive_failures = 0
    collection_success = 0
    collection_episodes = 0
    learning_success = 0
    learning_episodes = 0

    # 定义 Rich 进度条和显示面板
    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=None),
        TaskProgressColumn(),
        MofNCompleteColumn(),
        TimeRemainingColumn(),
        expand=True
    )
    task = progress.add_task("[green]Training SAC", total=total_episodes)

    # 定义监控信息表格
    def create_info_table(ep_reward, avg_reward, ep_steps, succ_rate, q_val, entropy, reason, dist):
        table = Table(title="UR5 SAC Training Dashboard", show_header=True, header_style="bold magenta")
        table.add_column("Metric", style="dim", width=15)
        table.add_column("Value", width=25)
        
        # 结果颜色
        reason_color = "white"
        if reason == 'success': reason_color = "green"
        elif reason == 'collision': reason_color = "red"
        elif reason == 'timeout': reason_color = "yellow"

        table.add_row("Episode", f"{len(episode_rewards)}")
        table.add_row("Total Steps", f"{total_steps}")
        table.add_row("Current Reward", f"{ep_reward:.2f}")
        table.add_row("Avg Reward (50)", f"{avg_reward:.2f}")
        table.add_row("Episode Steps", f"{ep_steps}")
        table.add_row("Success Rate", f"{succ_rate:.1f}%")
        if collection_episodes > 0:
            table.add_row("Coll. Succ Rate", f"{(collection_success / collection_episodes) * 100:.1f}% ({collection_success}/{collection_episodes})")
        if learning_episodes > 0:
            table.add_row("Learn Succ Rate", f"{(learning_success / learning_episodes) * 100:.1f}% ({learning_success}/{learning_episodes})")
        if consecutive_failures >= 3:
            table.add_row("Consec Fails", f"[red]{consecutive_failures}[/red]")
        table.add_row("Q Value", f"{q_val if isinstance(q_val, str) else f'{q_val:.2f}'}")
        table.add_row("Actor std", f"{entropy if isinstance(entropy, str) else f'{entropy:.4f}'}")
        table.add_row("Final Distance", f"{dist:.4f} m")
        table.add_row("Finish Reason", f"[{reason_color}]{reason}[/{reason_color}]")
        
        return table

    console.print(Panel(f"开始 SAC 强化学习训练流程\nTotal Episodes: {total_episodes}\nLearning Starts: {learning_starts}", title="Initialization", border_style="cyan"))

    try:
        with Live(console=console, refresh_per_second=4) as live:
            for episode in range(total_episodes):
                # 环境重置 - 传入课程学习进度
                curriculum_progress = min(1.0, episode / total_episodes)  # 0 → 1
                obs, info = env.reset(options={'progress': curriculum_progress})
                initial_dist = obs[-1]  # reset 后立即存下初始距离
                episode_reward = 0
                ep_steps = 0
                actions_in_episode = []
                
                for step in range(env.max_episode_steps):
                    prev_obs = obs
                    if total_steps < learning_starts:
                        action = env.action_space.sample() * COLLECTION_ACTION_LIMIT
                    else:
                        action, _ = model.predict(obs, deterministic=False)
                        if consecutive_failures >= 8:
                            noise = min(1.0, (consecutive_failures - 7) / 20.0) * CustomActor.ACTION_LIMIT * 0.4
                            action = action + np.random.normal(0, noise, 6).astype(np.float32)
                            action = np.clip(action, -CustomActor.ACTION_LIMIT, CustomActor.ACTION_LIMIT)
                    
                    actions_in_episode.append(action)
                    
                    next_obs, reward, terminated, truncated, info = env.step(action)
                    done = terminated or truncated
                    
                    model.replay_buffer.add(obs, next_obs, action, reward, terminated, [info])
                    
                    obs = next_obs
                    episode_reward += reward
                    total_steps += 1
                    ep_steps += 1
                    
                    if total_steps >= learning_starts and total_steps % train_freq == 0:
                        model.train(gradient_steps=gradient_steps, batch_size=batch_size)
                    
                    if done:
                        break
                
                # 统计
                episode_rewards.append(episode_reward)
                if info.get('success', False): 
                    success_count += 1
                if info.get('collision', False): 
                    collision_count += 1

                ep_started_in_collection = (total_steps - ep_steps) < learning_starts
                if ep_started_in_collection:
                    collection_episodes += 1
                    if info.get('success', False):
                        collection_success += 1
                else:
                    learning_episodes += 1
                    if info.get('success', False):
                        learning_success += 1

                if info.get('success', False):
                    consecutive_failures = 0
                else:
                    consecutive_failures += 1
                    if consecutive_failures > 0 and consecutive_failures % 5 == 0:
                        console.print(f"[yellow]连续失败 {consecutive_failures} 轮，探索噪声加大[/yellow]")
                
                final_dist = info.get('distance', 0)
                final_distances.append(final_dist)
                
                # 计算并记录改善量
                improvement = initial_dist - final_dist
                
                # 监控动作幅度
                if actions_in_episode:
                    action_std = np.std(actions_in_episode)
                    action_std_history.append(action_std)
                    if episode < 100 and total_steps < 10000:
                        act_limit = COLLECTION_ACTION_LIMIT if (total_steps - ep_steps) < learning_starts else CustomActor.ACTION_LIMIT
                        if action_std > act_limit * 0.4:
                            console.print(f"[cyan]Episode {episode+1}: action.std()={action_std:.4f}，探索正常[/cyan]")
                        else:
                            console.print(f"[yellow]Episode {episode+1}: action.std()={action_std:.4f}，探索不足![/yellow]")
                
                avg_reward = np.mean(episode_rewards[-50:]) if episode_rewards else 0
                success_rate = (success_count / (episode + 1)) * 100
                
                # 提取 SAC 内部参数 (Actor std 与 Q值)
                entropy = "N/A"
                q_val = "N/A"
                if total_steps >= learning_starts:
                    try:
                        with torch.no_grad():
                            obs_tensor = model.policy.obs_to_tensor(prev_obs)[0]
                            mean_actions, log_std, _ = model.actor.get_action_dist_params(obs_tensor)
                            std = torch.exp(log_std).mean().item()
                            entropy = std * CustomActor.ACTION_LIMIT  # 缩放后的实际动作标准差
                            action_tensor = torch.tensor(action).view(1, -1).to(model.device)
                            q1, q2 = model.critic(obs_tensor, action_tensor)
                            q_val = min(q1.item(), q2.item())
                    except Exception as e:
                        if episode % 50 == 0:
                            console.print(f"[yellow]Q/Std 计算失败: {str(e)}[/yellow]")

                # 更新 Live 显示内容
                reason = info.get('reason', 'unknown')
                dist = info.get('distance', 0)
                
                # 创建布局
                from rich.layout import Layout
                layout = Layout()
                layout.split_column(
                    Layout(progress, name="progress", size=3),
                    Layout(create_info_table(episode_reward, avg_reward, ep_steps, success_rate, q_val, entropy, reason, dist), name="info")
                )
                
                live.update(layout)
                progress.update(task, advance=1)
                
                # 输出每回合信息
                res_color = "white"
                if reason == 'success': 
                    res_color = "green"
                elif reason == 'collision': 
                    res_color = "red"
                elif reason == 'timeout': 
                    res_color = "yellow"
                
                console.log(
                    f"[blue]Ep {episode+1:d}/{total_episodes}[/blue] | "
                    f"[magenta]Rwd: {episode_reward:7.2f}[/magenta] | "
                    f"[green]Avg(50): {avg_reward:6.2f}[/green] | "
                    f"[{res_color}]Result: {reason:9s}[/{res_color}] | "
                    f"[cyan]Dist: {dist:.4f}m[/cyan] | "
                    f"[white]Steps: {ep_steps:3d}[/white] | "
                    f"[yellow]Success: {success_rate:.1f}%[/yellow] | "
                    f"[purple]Improve: {improvement:.4f}m[/purple]"
                )

        # 保存训练模型
        final_model_path = os.path.join(model_dir, "sac_ur5_final")
        model.save(final_model_path)
        print(Fore.GREEN + Style.BRIGHT + f"\n训练完成！最终成功率: {success_rate:.2f}% | 碰撞总数: {collision_count}")
        print(f"模型保存至: {final_model_path}")

    except KeyboardInterrupt:
        print(Fore.RED + "\n训练被用户中断，保存当前进度...")
        model.save(os.path.join(model_dir, "sac_ur5_interrupted"))
    
    finally:
        env.close()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == "__main__":
    main()
