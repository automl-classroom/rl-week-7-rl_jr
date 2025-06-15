"""
Deep Q-Learning with RND implementation.
"""

from typing import Any, Dict, List, Tuple

import gymnasium as gym
import hydra
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from omegaconf import DictConfig
from rl_exercises.week_4.dqn import DQNAgent, set_seed

from minigrid.wrappers import FlatObsWrapper

class RNDNetwork(nn.Module):

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, n_layers: int):
        super().__init__()
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())

        for _ in range(1, n_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)



class RNDDQNAgent(DQNAgent):
    """
    Deep Q-Learning agent with ε-greedy policy and target network.

    Derives from AbstractAgent by implementing:
      - predict_action
      - save / load
      - update_agent
    """

    def __init__(
        self,
        env: gym.Env,
        buffer_capacity: int = 10000,
        batch_size: int = 32,
        lr: float = 1e-3,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_final: float = 0.01,
        epsilon_decay: int = 500,
        target_update_freq: int = 1000,
        seed: int = 0,
        rnd_hidden_size: int = 128,
        rnd_lr: float = 1e-3,
        rnd_update_freq: int = 1000,
        rnd_n_layers: int = 2,
        rnd_reward_weight: float = 0.1,
    ) -> None:
        """
        Initialize replay buffer, Q-networks, optimizer, and hyperparameters.

        Parameters
        ----------
        env : gym.Env
            The Gym environment.
        buffer_capacity : int
            Max experiences stored.
        batch_size : int
            Mini-batch size for updates.
        lr : float
            Learning rate.
        gamma : float
            Discount factor.
        epsilon_start : float
            Initial ε for exploration.
        epsilon_final : float
            Final ε.
        epsilon_decay : int
            Exponential decay parameter.
        target_update_freq : int
            How many updates between target-network syncs.
        seed : int
            RNG seed.
        """
        super().__init__(
            env,
            buffer_capacity,
            batch_size,
            lr,
            gamma,
            epsilon_start,
            epsilon_final,
            epsilon_decay,
            target_update_freq,
            seed,
        )
        self.seed = seed
        # TODO: initialize the RND networks

        # try:
        #     input_dim = env.observation_space.shape[0]
        # except IndexError:
        #     input_dim = env.observation_space.n
        input_dim = env.observation_space.shape[0]
        output_dim = 32  # random length of feature vector

        self.predictor_rnd = RNDNetwork(input_dim, rnd_hidden_size, output_dim, rnd_n_layers)
        self.target_rnd = RNDNetwork(input_dim, rnd_hidden_size, output_dim, rnd_n_layers)

        for param in self.target_rnd.parameters():  # freeze target-network
            param.requires_grad = False

        self.rnd_optimizer = optim.Adam(self.predictor_rnd.parameters(), lr=rnd_lr)

        self.rnd_update_freq = rnd_update_freq
        self.rnd_reward_weight = rnd_reward_weight


    def update_rnd(
        self, training_batch: List[Tuple[Any, Any, float, Any, bool, Dict]]
    ) -> float:
        """
        Perform one gradient update on the RND network on a batch of transitions.

        Parameters
        ----------
        training_batch : list of transitions
            Each is (state, action, reward, next_state, done, info).
        """
        # TODO: get states and next_states from the batch
        next_states = [t[3] for t in training_batch]
        s_next = torch.tensor(np.array(next_states), dtype=torch.float32)

        # TODO: compute the MSE
        with torch.no_grad():
            target = self.target_rnd(s_next)
        pred = self.predictor_rnd(s_next)
        loss = nn.MSELoss()(pred, target)

        # TODO: update the RND network
        self.rnd_optimizer.zero_grad()
        loss.backward()
        self.rnd_optimizer.step()


    def get_rnd_bonus(self, state: np.ndarray) -> float:
        """Compute the RND bonus for a given state.

        Parameters
        ----------
        state : np.ndarray
            The current state of the environment.

        Returns
        -------
        float
            The RND bonus for the state.
        """
        # TODO: predict embeddings
        s = torch.tensor(state, dtype=torch.float32)
        with torch.no_grad():
            target = self.target_rnd(s)
            pred = self.predictor_rnd(s)

        # TODO: get error
        loss = nn.MSELoss()(pred, target)
        return loss.item() * self.rnd_reward_weight

    def train(self, num_frames: int, eval_interval: int = 1000, bin_size: int = 1000) -> None:
        """
        Run a training loop for a fixed number of frames.

        Parameters
        ----------
        num_frames : int
            Total environment steps.
        eval_interval : int
            Every this many episodes, print average reward.
        """
        state, _ = self.env.reset()
        ep_reward = 0.0
        recent_rewards: List[float] = []
        episode_rewards = []
        steps = []

        batch = None

        for frame in range(1, num_frames + 1):
            action = self.predict_action(state)
            next_state, reward, done, truncated, _ = self.env.step(action)

            # TODO: apply RND bonus
            reward += self.get_rnd_bonus(state)

            # store and step
            self.buffer.add(state, action, reward, next_state, done or truncated, {})
            state = next_state
            ep_reward += reward

            # update if ready
            if len(self.buffer) >= self.batch_size:
                batch = self.buffer.sample(self.batch_size)
                _ = self.update_agent(batch)

            if self.total_steps % self.rnd_update_freq == 0 and batch is not None:
                self.update_rnd(batch)

            if done or truncated:
                state, _ = self.env.reset()
                recent_rewards.append(ep_reward)
                episode_rewards.append(ep_reward)
                steps.append(frame)
                ep_reward = 0.0
                # logging
                if len(recent_rewards) % 10 == 0:
                    avg = np.mean(recent_rewards[-10:])
                    print(
                        f"Frame {frame}, AvgReward(10): {avg:.2f}, ε={self.epsilon():.3f}"
                    )

        # Saving to .csv for simplicity
        # Could also be e.g. npz
        print("Training complete.")
        training_data = pd.DataFrame({"steps": steps, "rewards": episode_rewards, "bin": [s // bin_size for s in steps]})
        training_data.to_csv(f"training_data_RND_seed_{self.seed}.csv", index=False)


@hydra.main(config_path="../configs/agent/", config_name="dqn", version_base="1.1")
def main(cfg: DictConfig):
    from gymnasium.wrappers import FlattenObservation

    for seed in cfg.seeds:
        # 1) build env
        env = gym.make(cfg.env.name, continuous=False)
        # env = gym.make(cfg.env.name, render_mode="human", continuous=False)

        env = FlattenObservation(env)
        # env = FlatObsWrapper(env)
        set_seed(env, seed)

        # 3) TODO: instantiate & train the agent
        agent = RNDDQNAgent(
            env=env,
            buffer_capacity=cfg.agent.buffer_capacity,
            batch_size=cfg.agent.batch_size,
            lr=cfg.agent.learning_rate,
            gamma=cfg.agent.gamma,
            epsilon_start=cfg.agent.epsilon_start,
            epsilon_final=cfg.agent.epsilon_final,
            epsilon_decay=cfg.agent.epsilon_decay,
            target_update_freq=cfg.agent.target_update_freq,
            seed=seed,
        )

        agent.train(cfg.train.num_frames, cfg.train.eval_interval)



if __name__ == "__main__":
    main()
