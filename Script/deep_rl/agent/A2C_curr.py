#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

from ..network import *
from ..component import *
from .BaseAgent import *

class A2CAgentCurr(BaseAgent):
    def __init__(self, config):
        BaseAgent.__init__(self, config)
        self.config = config
        self.task = config.task_fn()
        self.network = config.network_fn()
        self.optimizer = config.optimizer_fn(self.network.parameters())
        self.total_steps = 0
        self.states = self.task.reset()
        self.step_rewards = 0
        self.roll_rewards = 0
        self.eval_rewards = 0

    def eval(self, sub_goal):
        config = self.config
        states = self.states
        self.eval_rewards = 0
        for _ in range(config.rollout_length):
            actions, log_probs, entropy, values = self.network(config.state_normalizer(states), sub_goal)
            next_states, rewards, terminals, _ = self.task.step(actions.detach().cpu().numpy(), rollout=True)
            self.eval_rewards += rewards
            states = next_states.view(1,3,84,84).cuda()
            if terminals:
                break
        self.states = states

    # rollout subpolicy
    def roll(self, sub_goal):
        config = self.config
        states = self.states
        self.roll_rewards = 0

        for _ in range(config.rollout_length):
            actions, log_probs, entropy, values = self.network(config.state_normalizer(states), sub_goal)
            next_states, rewards, terminals, _ = self.task.step(actions.detach().cpu().numpy(), sub_goal, True)
            self.roll_rewards += rewards
            states = next_states.view(1,3,84,84).cuda()
            if terminals:
                break
        self.states = states

    # update subpolicy
    def step(self, sub_goal):
        config = self.config
        rollout = []
        states = self.states       
        self.step_rewards = 0
        for _ in range(config.step_length):
            actions, log_probs, entropy, values = self.network(config.state_normalizer(states), sub_goal)
            next_states, rewards, terminals, _ = self.task.step(actions.detach().cpu().numpy(), sub_goal)
            self.step_rewards += rewards
            rollout.append([log_probs, values, actions, rewards, 1 - terminals, entropy])
            states = next_states.view(1,3,84,84).cuda()
            if terminals:
                break

        self.states = states
        pending_value = self.network(config.state_normalizer(states), sub_goal)[-1]
        rollout.append([None, pending_value, None, None, None, None])

        processed_rollout = [None] * (len(rollout) - 1)
        advantages = tensor(np.zeros((1, 1)))
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

        steps = config.rollout_length
        self.total_steps += steps

        return total_loss

    def save(self, filename):
        torch.save(self.network.state_dict(), filename)
        with open(filename[0:-4], 'wb') as f:
            pickle.dump([self.total_steps, self.optimizer, self.states], f)

    def load(self, filename):
        with open(filename[0:-4], 'rb') as f:
            a = pickle.load(f)
        self.total_steps = a[0]
        self.optimizer = a[1]
        self.states = a[2]

        state_dict = torch.load(filename, map_location=lambda storage, loc: storage)
        self.network.load_state_dict(state_dict)
