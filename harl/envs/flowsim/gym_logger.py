from harl.common.base_logger import BaseLogger
from torch.utils.tensorboard import SummaryWriter
import time
import numpy as np
from harl.envs.flowsim.flowsim import FlowSimEnv
from pathlib import Path

class FlowSimLogger(BaseLogger):
    def __init__(self, args, algo_args, env_args, num_agents, writer : SummaryWriter, run_dir):
        super().__init__(args, algo_args, env_args, num_agents, writer, run_dir)
        self.best_reward = -np.inf

    def get_task_name(self):
        return self.env_args["scenario"]
    
    def per_step(self, data):
        (
            obs,
            share_obs,
            rewards,
            dones,
            infos,
            available_actions,
            values,
            actions,
            action_log_probs,
            rnn_states,
            rnn_states_critic,
        ) = data

        best_rew_idx = np.argmax(rewards, axis=0)[0] 
        best_rew = rewards[best_rew_idx][0]

        if best_rew > self.best_reward:
            self.best_reward = best_rew
            best_env : FlowSimEnv = infos[best_rew_idx][0]['env']['graph_env_inst']
            inital_output_path = Path(self.run_dir, "outputs","initial_output_plans.xml")
            if not inital_output_path.exists():
                # Save initial output for comparison
                best_env.save_plans_from_flow_res(inital_output_path.absolute())
                best_env.dataset.save_clusters(Path(self.run_dir, "outputs","clusters.txt").absolute())
            best_env.save_plans_from_flow_res(Path(self.run_dir, "outputs","best_output.xml").absolute())
    
    def episode_log(
        self, actor_train_infos, critic_train_info, actor_buffer, critic_buffer
    ):

        """Log information for each episode."""
        self.total_num_steps = (
            self.episode
            * self.algo_args["train"]["episode_length"]
            * self.algo_args["train"]["n_rollout_threads"]
        )
        self.end = time.time()
        print(
            "Env {} Task {} Algo {} Exp {} updates {}/{} episodes, total num timesteps {}/{}, FPS {}.".format(
                self.args["env"],
                self.task_name,
                self.args["algo"],
                self.args["exp_name"],
                self.episode,
                self.episodes,
                self.total_num_steps,
                self.algo_args["train"]["num_env_steps"],
                int(self.total_num_steps / (self.end - self.start)),
            )
        )

        avg_step_rewards = critic_buffer.rewards.mean()
        critic_train_info["average_step_rewards"] = critic_buffer.rewards.mean()

        print(
            "Average step reward is {}.".format(
                critic_train_info["average_step_rewards"]
            )
        )

        if len(self.done_episodes_rewards) > 0:
            aver_episode_rewards = np.mean(self.done_episodes_rewards)
            print(
                "Some episodes done, average episode reward is {}.\n".format(
                    aver_episode_rewards
                )
            )
            self.writer.add_scalars(
                "train_episode_rewards",
                {"aver_rewards": aver_episode_rewards},
                self.total_num_steps,
            )
            self.done_episodes_rewards = []

        self.writer.add_scalar("avg_reward/step", avg_step_rewards, self.total_num_steps)
        self.writer.add_scalar("value_loss", critic_train_info["value_loss"], self.total_num_steps)
        self.writer.add_scalar("value_loss", self.best_reward, self.total_num_steps)

