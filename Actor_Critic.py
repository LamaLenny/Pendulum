import torch
import torch.nn as nn
import numpy as np
from torch.distributions import Normal
import gym
import random
from collections import deque
import torch.nn.functional as F
from Models import Actor, Critic

GAMMA = 0.93
LR_ACTOR = 1e-4
LR_CRITIC = 1e-3
ENTROPY_COEF = 1e-1

seed = 39
seed_to_test = 21
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

"""# Actor Critic"""


class AC:
    def __init__(self, state_dim):
        self.actor = Actor(state_dim).to(device)
        self.critic = Critic(state_dim).to(device)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=LR_ACTOR)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=LR_CRITIC)

    def update(self, data):
        states, next_states, rewards, dones, log_probs = zip(*data)
        states = torch.from_numpy(np.array(states)).float().to(device)
        next_states = torch.from_numpy(np.array(next_states)).float().to(device)
        log_probs = torch.cat(log_probs)
        rewards = torch.from_numpy(np.array(rewards)).float().to(device).unsqueeze(1)
        dones = torch.from_numpy(np.array(dones)).to(device).unsqueeze(1)

        V = self.critic(states)
        Q = rewards + GAMMA * self.critic(next_states).detach() * (~dones)

        ################ Actor Update ###################
        actor_loss = ((Q - V.detach()).T @ log_probs)
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        ################ Critic Update ###################
        critic_loss = F.mse_loss(Q, V)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

    def act(self, state):
        mu, sigma = self.actor(torch.from_numpy(np.array(state)).float().unsqueeze(0).to(device))
        distribution = Normal(mu, sigma)
        action = distribution.sample()
        log_prob = distribution.log_prob(action)
        return np.clip(action.cpu().numpy(), -2, 2), log_prob

    def save(self, i, num_repeat=200):
        env_ = gym.make("Pendulum-v0")
        env_.seed(seed_to_test)
        rews = np.zeros(shape=(num_repeat, 1))
        for k in range(num_repeat):
            state = env_.reset()
            done = False
            while not done:
                action, _ = self.act(state)
                state, reward, done, _ = env_.step(action)
                rews[k] += reward
        mean = np.mean(rews)
        std = np.std(rews)
        s = "agent" + str(i) + ".pickle"
        torch.save(self.actor.state_dict(), s)
        return mean, std


"""# Обучение"""

if __name__ == '__main__':
    env = gym.make('Pendulum-v0')
    env.seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    agent = AC(3)
    episodes = 30000
    m = None
    last_good = (None, 1000)
    for i in range(episodes):
        state = env.reset()
        done = False
        data = []
        episode_reward = 0
        while not done:
            action, log_prob = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            data.append((state, next_state, reward / 10, done, log_prob))
            state = next_state
            episode_reward += reward
        agent.update(data)
        if not i % 500:
            mean, std = agent.save(i, 200)
            print(f"I did {i}th episode. Mean and std (200 random episodes): {mean, std}")
