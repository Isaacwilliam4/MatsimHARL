from harl.common.base_logger import BaseLogger
from torch.utils.tensorboard import SummaryWriter
import time
import numpy as np
from rlevmatsim.envs.ocp.rl_ocp_env import RLOCPEnv
from pathlib import Path
import torch
from copy import deepcopy as dc

class OCPLogger(BaseLogger):
    def __init__(self, args, algo_args, env_args, num_agents, writer : SummaryWriter, run_dir):
        super().__init__(args, algo_args, env_args, num_agents, writer, run_dir)
        self.run_dir = run_dir
        self.best_reward = -np.inf
        env_args["dataset"].save_clusters(self.run_dir)
        self.saved_initial_output = False

    def get_task_name(self):
        return "ocp"
    
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


        if not self.saved_initial_output:
            env : RLOCPEnv = infos[0][0]['graph_env_inst']
            env.dataset.save_output(self.run_dir, "initial_output")
            self.saved_initial_output = True

        best_rew_idx = np.argmax(rewards, axis=0)[0]
        best_rew = rewards[best_rew_idx][0]
        self.avg_reward = np.mean(rewards[:,0])
        best_env : RLOCPEnv = infos[best_rew_idx][0]['graph_env_inst']
        self.charger_cost = best_env.dataset.charger_cost
        self.num_dynamic_chargers = best_env.dataset.linegraph.x[:,-1].sum()
        self.num_static_chargers = best_env.dataset.linegraph.x[:,-2].sum()
        self.charger_model_loss = best_env.dataset.charger_model_loss
        self.charge_efficiency = best_env.charge_reward

        if best_rew > self.best_reward:
            self.best_reward = best_rew
            self.best_env = best_env
            self.best_env.dataset.save_charger_config_to_csv(self.run_dir, best_rew)
            self.best_env.dataset.save_output(self.run_dir, "best_output")
            with open(Path(self.run_dir) / "matsim_charge_model.pt", "wb") as f:
                torch.save(self.best_env.dataset.charge_model, f)
    
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

        critic_train_info["average_step_rewards"] = critic_buffer.rewards.mean()

        print(
            "Average step reward is {}.".format(
                critic_train_info["average_step_rewards"]
            )
        )

        self.writer.add_scalar("best_reward", self.best_reward, self.total_num_steps)
        self.writer.add_scalar("avg_reward", self.avg_reward, self.total_num_steps)
        self.writer.add_scalar("charger_cost", self.charger_cost, self.total_num_steps)
        self.writer.add_scalar("num_static_chargers", self.num_static_chargers, self.total_num_steps)
        self.writer.add_scalar("num_dynamic_chargers", self.num_dynamic_chargers, self.total_num_steps)
        self.writer.add_scalar("charger_model_loss", self.charger_model_loss, self.total_num_steps)
        self.writer.add_scalar("charge_efficiency", self.charge_efficiency, self.total_num_steps)

        # only log the first agent for performance reasons
        for k, v in actor_train_infos[0].items():
            agent_k = "agent%i/" % 0 + k
            self.writer.add_scalars(agent_k, {agent_k: v}, self.total_num_steps)

        for k, v in critic_train_info.items():
            critic_k = "critic/" + k
            self.writer.add_scalars(critic_k, {critic_k: v}, self.total_num_steps)
