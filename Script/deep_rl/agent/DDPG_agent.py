#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

from ..network import *
from ..component import *
from .BaseAgent import *
import torchvision
import pickle

class DDPGAgent(BaseAgent):
    def __init__(self, config):
        BaseAgent.__init__(self, config)
        self.config = config
        self.task = config.task_fn()
        self.network = config.network_fn()
        self.target_network = config.network_fn()
        self.target_network.load_state_dict(self.network.state_dict())
        self.replay = config.replay_fn()
        self.random_process = config.random_process_fn()
        self.total_steps = 0
        self.state = None
        self.episode_reward = 0
        self.episode_rewards = []

    def soft_update(self, target, src):
        for target_param, param in zip(target.parameters(), src.parameters()):
            target_param.detach_()
            target_param.copy_(target_param * (1.0 - self.config.target_network_mix) +
                                    param * self.config.target_network_mix)

    def eval_step(self, state):
        self.config.state_normalizer.set_read_only()
        state = np.stack([self.config.state_normalizer(state)])
        action = self.network(state)
        self.config.state_normalizer.unset_read_only()
        return to_np(action).flatten()

    def step(self):
        config = self.config
        if self.state is None:
            self.random_process.reset_states()
            self.state = self.task.reset()
            self.state = config.state_normalizer(self.state)
        action = self.network(np.stack([self.state]))
        action = to_np(action).flatten()
        action += self.random_process.sample()
        next_state, reward, done, info = self.task.step(action)
        next_state = self.config.state_normalizer(next_state)
        self.episode_reward += reward
        reward = self.config.reward_normalizer(reward)
        self.replay.feed([self.state.data.cpu().numpy(), action, reward, next_state.data.cpu().numpy(), int(done)])
        if done:
            next_state = None
            self.episode_rewards.append(self.episode_reward)
            self.episode_reward = 0
        self.state = next_state
        self.total_steps += 1

        loss = 0

        if self.replay.size() >= config.min_memory_size:
            experiences = self.replay.sample()
            states, actions, rewards, next_states, terminals = experiences

            phi_next = self.target_network.feature(next_states)
            a_next = self.target_network.actor(phi_next)
            q_next = self.target_network.critic(phi_next, a_next)
            terminals = tensor(terminals).unsqueeze(1)
            rewards = tensor(rewards).unsqueeze(1)
            q_next = config.discount * q_next * (1 - terminals)
            q_next.add_(rewards)
            q_next = q_next.detach()
            phi = self.network.feature(states)
            q = self.network.critic(phi, tensor(actions))
            critic_loss = (q - q_next).pow(2).mul(0.5).sum(-1).mean()
            loss += critic_loss

            self.network.zero_grad()
            critic_loss.backward()
            self.network.critic_opt.step()

            phi = self.network.feature(states)
            action = self.network.actor(phi)
            policy_loss = -self.network.critic(phi.detach(), action).mean()
            loss += policy_loss

            self.network.zero_grad()
            policy_loss.backward()
            self.network.actor_opt.step()

            self.soft_update(self.target_network, self.network)

        return loss

    def save(self, filename):
        torch.save(self.network.state_dict(), filename)
        torch.save(self.target_network.state_dict(), filename.strip(".bin")+"_target"+".bin")
        with open(filename.strip(".bin"), 'wb') as f:
            pickle.dump([self.random_process, self.replay, self.total_steps, self.state], f)

    def load(self, filename):
        state_dict = torch.load(filename, map_location=lambda storage, loc: storage)
        state_dict_target = torch.load(filename.strip(".bin")+"_target"+".bin", map_location=lambda storage, loc: storage)
        self.network.load_state_dict(state_dict)
        self.target_network.load_state_dict(state_dict_target)
        with open(filename.strip(".bin"), 'rb') as f:
            a = pickle.load(f)
        self.random_process = a[0]
        self.replay = a[1]
        self.total_steps = a[2]
        # self.state = a[3]