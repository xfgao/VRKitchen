#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

from ..network import *
from ..component import *
from ..utils import *
import time
from .BaseAgent import *

class DQNActor():
    def __init__(self, config):
        self.config = config
        self._state = None
        self._task = self.config.task_fn()
        self._network = None
        self._total_steps = 0

    def set_network(self, net):
        self._network = net

    def step(self):
        if self._state is None:
            self._state = self._task.reset()
        q_values = self._network(self.config.state_normalizer(self._state))
        q_values = to_np(q_values).flatten()
        if self._total_steps < self.config.exploration_steps \
                or np.random.rand() < self.config.random_action_prob():
            action = np.random.randint(0, len(q_values))
        else:
            action = np.argmax(q_values)
        next_state, reward, done, info = self._task.step(action)
        entry = [self._state, action, reward, next_state, int(done), info]
        self._total_steps += 1
        self._state = next_state
        return entry

class DQNAgent(BaseAgent):
    def __init__(self, config):
        BaseAgent.__init__(self, config)
        self.config = config

        self.replay = self.config.replay_fn()
        self.actor = DQNActor(self.config)

        self.network = self.config.network_fn()
        self.network.share_memory()
        self.target_network = self.config.network_fn()
        self.target_network.load_state_dict(self.network.state_dict())
        self.optimizer = self.config.optimizer_fn(self.network.parameters())

        self.actor.set_network(self.network)

        self.episode_reward = 0
        self.episode_rewards = []

        self.total_steps = 0
        self.batch_indices = range_tensor(self.replay.batch_size)

    def close(self):
        close_obj(self.replay)
        close_obj(self.actor)

    def eval_step(self, state):
        self.config.state_normalizer.set_read_only()
        state = self.config.state_normalizer(state)
        q = self.network(state)
        action = np.argmax(to_np(q).flatten())
        self.config.state_normalizer.unset_read_only()
        return action

    def step(self):
        transitions = self.actor.step()
        experiences = []
        state, action, reward, next_state, done, _ = transitions
        state = state.squeeze().data.cpu().numpy()
        next_state = next_state.squeeze().data.cpu().numpy()
        self.episode_reward += reward
        self.total_steps += 1
        if done:
            self.episode_rewards.append(self.episode_reward)
            self.episode_reward = 0
        experiences.append([state, action, reward, next_state, done])
        self.replay.feed_batch(experiences)

        loss = 0

        if self.total_steps > self.config.exploration_steps:
            experiences = self.replay.sample()
            states, actions, rewards, next_states, terminals = experiences
            states = self.config.state_normalizer(states)
            next_states = self.config.state_normalizer(next_states)
            q_next = self.target_network(next_states).detach()
            if self.config.double_q:
                best_actions = torch.argmax(self.network(next_states), dim=-1)
                q_next = q_next[self.batch_indices, best_actions]
            else:
                q_next = q_next.max(1)[0]
            terminals = tensor(terminals)
            rewards = tensor(rewards)
            q_next = self.config.discount * q_next * (1 - terminals)
            q_next.add_(rewards)
            actions = tensor(actions).long()
            q = self.network(states)
            q = q[self.batch_indices, actions]
            loss = (q_next - q).pow(2).mul(0.5).mean()
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.network.parameters(), self.config.gradient_clip)
            self.optimizer.step()

        if self.total_steps / self.config.sgd_update_frequency % \
                self.config.target_network_update_freq == 0:
            self.target_network.load_state_dict(self.network.state_dict())

        return loss

    def save(self, filename):
        torch.save(self.network.state_dict(), filename)
        torch.save(self.target_network.state_dict(), filename.strip(".bin")+"_target"+".bin")
        with open(filename.strip(".bin"), 'wb') as f:
            pickle.dump([self.replay, self.total_steps, self.optimizer], f)

    def load(self, filename):
        state_dict = torch.load(filename, map_location=lambda storage, loc: storage)
        state_dict_target = torch.load(filename.strip(".bin")+"_target"+".bin", map_location=lambda storage, loc: storage)
        self.network.load_state_dict(state_dict)
        self.target_network.load_state_dict(state_dict_target)
        with open(filename.strip(".bin"), 'rb') as f:
            a = pickle.load(f)
        self.replay = a[0]
        self.total_steps = a[1]
        self.optimizer = a[2]
        self.actor.set_network(self.network)