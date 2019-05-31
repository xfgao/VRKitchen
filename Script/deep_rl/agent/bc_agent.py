from ..network import *
from ..component import *
from .BaseAgent import *
import torch.nn.functional as F

class BCAgent(BaseAgent):
	def __init__(self, config):
		BaseAgent.__init__(self, config)
		self.config = config
		self.task = config.task_fn()
		self.network = config.network_fn
		self.optimizer = config.optimizer_fn(self.network.parameters())
		self.total_steps = 0
		self.state = None
		self.episode_reward = 0
		self.episode_rewards = []

	def step(self):
		if self.state is None:
			self.state = self.task.reset()
		score = self.network(self.state)
		m = torch.distributions.categorical.Categorical(F.softmax(score, 1))
		# action = m.sample()
		action = torch.argmax(score, dim=1)
		next_state, reward, done, info = self.task.step(action)
		self.episode_reward += reward
		loss = 0
		loss = -m.log_prob(action) * reward
		# self.network.zero_grad()
		# loss.backward(retain_graph=True)
		# self.optimizer.step()
		if done:
			next_state = None
			self.episode_rewards.append(self.episode_reward)
			self.episode_reward = 0
			self.network.hidden = self.network.init_hidden()
		self.state = next_state
		self.total_steps += 1

		return loss
