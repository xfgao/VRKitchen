import numpy as np
import random
import time
from skimage import color, transform
from PIL import Image

from torchvision import transforms
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
import torch.optim as optim

from deep_rl import *
from DiscreteAgent import DiscreteAgent
from task_dish.Cut import Cut
from task_dish.Juice import Juice
from task_dish.Sandwich import Sandwich
from task_dish.Stew import Stew

TaskMap = {"Cut": Cut, "Sandwich": Sandwich, "Stew": Stew, "Juice": Juice}

def dqn_dish(task_name, mstep):
	config = Config()
	config.name = "dqn"
	config.history_length = 1
	config.task_fn = lambda : TaskMap[task_name](frame_skip=1, name=task_name, max_steps=mstep)
	config.task_name = config.task_fn().name
	config.eval_env = config.task_fn()
	config.state_dim = 84
	config.action_dim = config.task_fn().action_dim
	config.optimizer_fn = lambda params: torch.optim.RMSprop(
		params, lr=0.0001, alpha=0.95, eps=0.01, centered=True)
	config.network_fn = lambda: VanillaNet(config.action_dim, NatureConvBody(in_channels=3))
	# config.network_fn = lambda: DuelingNet(config.action_dim, NatureConvBody(in_channels=config.history_length))
	config.random_action_prob = LinearSchedule(1.0, 0.1, 3e5)
	config.replay_fn = lambda: Replay(memory_size=int(1e4), batch_size=32)
	config.batch_size = 32
	config.state_normalizer = ImageNormalizer()
	config.discount = 0.95
	config.target_network_update_freq = 30000
	config.exploration_steps = 90000
	config.sgd_update_frequency = 4
	config.gradient_clip = 5
	# config.double_q = True
	config.double_q = False
	config.max_steps = 1e7
	config.log_interval = 3000
	config.max_eps = 10450
	config.logger = get_logger()
	config.save_interval = 30000
	config.tag = "vanilla"
	my_agent = DQNAgent(config)
	try:
		my_agent.load('data/model-%s-%s-%s.bin' % (config.name, config.task_name, config.tag))
	except Exception as e:
		print(e)
	run_steps(my_agent)

def a2c_dish(task_name, mstep):
	config = Config()
	config.name = "a2c"
	config.history_length = 1
	config.num_workers = 1
	config.task_fn = lambda : TaskMap[task_name](frame_skip=1, name=task_name, max_steps=mstep)
	config.task_name = config.task_fn().name
	config.eval_env = config.task_fn()
	config.state_dim = 84
	config.action_dim = config.task_fn().action_dim
	config.optimizer_fn = lambda params: torch.optim.RMSprop(params, lr=1e-4, alpha=0.99, eps=1e-5)
	config.network_fn = lambda: CategoricalActorCriticNet(config.state_dim, config.action_dim, NatureConvBody(in_channels=3))
	config.state_normalizer = ImageNormalizer()
	config.discount = 0.95
	config.use_gae = True
	config.gae_tau = 1.0
	config.entropy_weight = 0.01
	config.rollout_length = 3000
	config.gradient_clip = 0.5
	config.max_steps = 1e7
	config.log_interval = 3000
	config.max_eps = 10450
	config.logger = get_logger()
	config.save_interval = 30000
	config.tag = "vanilla"
	my_agent = A2CAgent(config)
	try:
		my_agent.load('data/model-%s-%s-%s.bin' % (config.name, config.task_name, config.tag))
	except Exception as e:
		print(e)
	run_steps(my_agent)

def ppo_dish(task_name, mstep):
	config = Config()
	config.name = "ppo"
	config.history_length = 1
	config.task_fn = lambda : TaskMap[task_name](frame_skip=1, name=task_name, max_steps=mstep)
	config.task_name = config.task_fn().name
	config.num_workers = 1
	config.eval_env = config.task_fn()
	config.optimizer_fn = lambda params: torch.optim.RMSprop(params, lr=1e-4, alpha=0.99, eps=1e-5)
	config.state_dim = 84
	config.action_dim = config.task_fn().action_dim
	config.network_fn = lambda: CategoricalActorCriticNet(config.state_dim, config.action_dim, NatureConvBody(in_channels=3))
	config.state_normalizer = ImageNormalizer()
	config.discount = 0.95
	config.use_gae = True
	config.gae_tau = 1.0
	config.entropy_weight = 0.01
	config.gradient_clip = 0.5
	config.rollout_length = 3000
	config.optimization_epochs = 4
	config.num_mini_batches = 500
	config.ppo_ratio_clip = 0.1
	config.log_interval = 3000
	config.max_eps = 10450
	config.max_steps = 1e7
	config.logger = get_logger()
	config.save_interval = 30000
	config.tag = "vanilla"
	my_agent = PPOAgent(config)
	try:
		my_agent.load('data/model-%s-%s-%s.bin' % (config.name, config.task_name, config.tag))
	except Exception as e:
		print(e)
	run_steps(my_agent)

def run_rl():
	task_name = "Sandwich"
	max_steps = 1000
	ppo_dish(task_name, max_steps)
	# dqn_dish(task_name, max_steps)
	# a2c_dish(task_name, max_steps)

if __name__ == "__main__":
	run_rl()





