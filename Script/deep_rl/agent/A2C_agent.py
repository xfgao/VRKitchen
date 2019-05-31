#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

from ..network import *
from ..component import *
from .BaseAgent import *

class A2CAgent(BaseAgent):
    def __init__(self, config):
        BaseAgent.__init__(self, config)
        self.config = config
        self.task = config.task_fn()
        self.network = config.network_fn()
        self.optimizer = config.optimizer_fn(self.network.parameters())
        self.total_steps = 0
        self.states = self.task.reset()
        self.episode_rewards = []
        self.online_rewards = np.zeros(config.num_workers)
        self.terminals = []

    def step(self):
        config = self.config
        rollout = []
        states = self.states.view(1,3,84,84).cuda()
        for _ in range(config.rollout_length):
            actions, log_probs, entropy, values = self.network(config.state_normalizer(states))

            next_states, rewards, terminals, _ = self.task.step(actions.detach().cpu().numpy())
            self.online_rewards += rewards
            rewards = config.reward_normalizer(rewards)
            terminals = [terminals]
            for i, terminal in enumerate(terminals):
                if terminals[i]:
                    self.episode_rewards.append(self.online_rewards[i])
                    self.online_rewards[i] = 0

            rollout.append([log_probs, values, actions, rewards, 1 - terminals[0], entropy])
            states = next_states.view(1,3,84,84).cuda()

        self.states = states
        pending_value = self.network(config.state_normalizer(states))[-1]
        rollout.append([None, pending_value, None, None, None, None])

        processed_rollout = [None] * (len(rollout) - 1)
        advantages = tensor(np.zeros((config.num_workers, 1)))
        returns = pending_value.detach()
        for i in reversed(range(len(rollout) - 1)):
            log_prob, value, actions, rewards, terminals, entropy = rollout[i]
            terminals = [terminals]
            rewards = [rewards]
            terminals = tensor(terminals).unsqueeze(1)
            rewards = tensor(rewards).unsqueeze(1)
            next_value = rollout[i + 1][1]
            returns = rewards + config.discount * terminals * returns
            if not config.use_gae:
                advantages = returns - value.detach()
            else:
                td_error = rewards + config.discount * terminals * next_value.detach() - value.detach()
                advantages = advantages * config.gae_tau * config.discount * terminals + td_error
            processed_rollout[i] = [log_prob, value, returns, advantages, entropy]

        log_prob, value, returns, advantages, entropy = map(lambda x: torch.cat(x, dim=0), zip(*processed_rollout))
        policy_loss = -log_prob * advantages
        value_loss = 0.5 * (returns - value).pow(2)
        entropy_loss = entropy.mean()

        self.policy_loss = np.mean(policy_loss.cpu().detach().numpy())
        self.entropy_loss = np.mean(entropy_loss.cpu().detach().numpy())
        self.value_loss = np.mean(value_loss.cpu().detach().numpy())

        self.optimizer.zero_grad()
        
        total_loss = (policy_loss - config.entropy_weight * entropy_loss +
        config.value_loss_weight * value_loss).mean()

        (policy_loss - config.entropy_weight * entropy_loss +
         config.value_loss_weight * value_loss).mean().backward()
        # print(self.network.parameters)
        nn.utils.clip_grad_norm_(self.network.parameters(), config.gradient_clip)
        self.optimizer.step()

        steps = config.rollout_length * config.num_workers
        self.total_steps += steps

        return total_loss

    def save(self, filename):
        torch.save(self.network.state_dict(), filename)
        with open(filename.strip(".bin"), 'wb') as f:
            pickle.dump([self.total_steps, self.optimizer, self.online_rewards, self.episode_rewards, self.states], f)

    def load(self, filename):
        state_dict = torch.load(filename, map_location=lambda storage, loc: storage)
        self.network.load_state_dict(state_dict)
        with open(filename.strip(".bin"), 'rb') as f:
            a = pickle.load(f)
        self.total_steps = a[0]
        self.optimizer = a[1]
        self.online_rewards = a[2]
        self.episode_rewards = a[3]
        self.states = a[4]