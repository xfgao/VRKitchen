#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

from ..network import *
from ..component import *
from .BaseAgent import *

class PPOAgent(BaseAgent):
    def __init__(self, config):
        BaseAgent.__init__(self, config)
        self.config = config
        self.task = config.task_fn()
        self.network = config.network_fn()
        self.opt = config.optimizer_fn(self.network.parameters())
        self.total_steps = 0
        self.online_rewards = np.zeros(config.num_workers)
        self.episode_rewards = []
        self.states = self.task.reset()
        self.states = config.state_normalizer(self.states)

    def step(self):
        config = self.config
        rollout = []
        states = torch.stack([self.states]).view(1,3,84,84).cuda()
        loss = 0
        for _ in range(config.rollout_length):
            actions, log_probs, _, values = self.network(states)
            next_states, rewards, terminals, _ = self.task.step(actions.cpu().detach().numpy())
            self.online_rewards += rewards
            rewards = config.reward_normalizer(rewards)
            terminals = [terminals]
            for i, terminal in enumerate(terminals):
                if terminals[i]:
                    self.episode_rewards.append(self.online_rewards[i])
                    self.online_rewards[i] = 0
            next_states = config.state_normalizer(next_states)
            rollout.append([states, values.detach(), actions.detach(), log_probs.detach(), rewards, 1 - terminals[0]])
            states = next_states.view(1,3,84,84).cuda()

        self.states = states
        pending_value = self.network(states)[-1]
        rollout.append([states, pending_value, None, None, None, None])

        processed_rollout = [None] * (len(rollout) - 1)
        advantages = tensor(np.zeros((config.num_workers, 1)))
        returns = pending_value.detach()
        for i in reversed(range(len(rollout) - 1)):
            states, value, actions, log_probs, rewards, terminals = rollout[i]
            terminals = [terminals]
            rewards = [rewards]
            terminals = tensor(terminals).unsqueeze(1)
            rewards = tensor(rewards).unsqueeze(1)
            actions = tensor(actions)
            states = tensor(states)
            next_value = rollout[i + 1][1]
            returns = rewards + config.discount * terminals * returns
            if not config.use_gae:
                advantages = returns - value.detach()
            else:
                td_error = rewards + config.discount * terminals * next_value.detach() - value.detach()
                advantages = advantages * config.gae_tau * config.discount * terminals + td_error
            processed_rollout[i] = [states, actions, log_probs, returns, advantages]

        states, actions, log_probs_old, returns, advantages = map(lambda x: torch.cat(x, dim=0), zip(*processed_rollout))
        advantages = (advantages - advantages.mean()) / advantages.std()

        batcher = Batcher(states.size(0) // config.num_mini_batches, [np.arange(states.size(0))])
        for _ in range(config.optimization_epochs):
            batcher.shuffle()
            while not batcher.end():
                batch_indices = batcher.next_batch()[0]
                batch_indices = tensor(batch_indices).long()
                sampled_states = states[batch_indices]
                sampled_actions = actions[batch_indices]
                sampled_log_probs_old = log_probs_old[batch_indices]
                sampled_returns = returns[batch_indices]
                sampled_advantages = advantages[batch_indices]
                _, log_probs, entropy_loss, values = self.network(sampled_states, sampled_actions)
                ratio = (log_probs - sampled_log_probs_old).exp()
                obj = ratio.cpu() * sampled_advantages.cpu()
                obj_clipped = ratio.clamp(1.0 - self.config.ppo_ratio_clip,
                                          1.0 + self.config.ppo_ratio_clip).cpu() * sampled_advantages.cpu()
                policy_loss = -torch.min(obj, obj_clipped).mean(0) - config.entropy_weight * entropy_loss.cpu().mean()

                value_loss = 0.5 * (sampled_returns - values).pow(2).mean()

                self.opt.zero_grad()
                loss += policy_loss.cpu() + value_loss.cpu()
                (policy_loss.cpu()+value_loss.cpu()).backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), config.gradient_clip)
                self.opt.step()

        steps = config.rollout_length * config.num_workers
        self.total_steps += steps
        loss /= config.optimization_epochs
        return loss

    def save(self, filename):
        torch.save(self.network.state_dict(), filename)
        with open(filename.strip(".bin"), 'wb') as f:
            pickle.dump([self.total_steps, self.opt, self.online_rewards, self.episode_rewards, self.states], f)

    def load(self, filename):
        state_dict = torch.load(filename, map_location=lambda storage, loc: storage)
        self.network.load_state_dict(state_dict)
        with open(filename.strip(".bin"), 'rb') as f:
            a = pickle.load(f)
        self.total_steps = a[0]
        self.opt = a[1]
        self.online_rewards = a[2]
        self.episode_rewards = a[3]
        self.states = a[4]