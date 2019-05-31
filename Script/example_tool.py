import numpy as np
import torch
import random

from DiscreteAgent import DiscreteAgent
from deep_rl import *
from task_tool import CutCarrot, PourWater, GetWater, OpenCan, PeelKiwi

# torch.manual_seed(0)
# np.random.seed(0)
# random.seed(0)

TaskMap = {"CutCarrot":CutCarrot, "PourWater":PourWater, "GetWater":GetWater, "OpenCan":OpenCan, "PeelKiwi":PeelKiwi}

def ddpg_pixel(task_name, mstep):
	config = Config()
	config.task_fn = lambda : TaskMap[task_name](frame_skip=1, max_steps=mstep)
	config.eval_env = config.task_fn()
	config.state_dim = 84
	config.action_dim = 7
	config.task_name = config.task_fn().name
	config.name = "ddpg"
	phi_body = NatureConvBody()
	config.network_fn = lambda: DeterministicActorCriticNet(
		config.state_dim, config.action_dim, phi_body=phi_body,
		actor_body=FCBody(phi_body.feature_dim, (50, ), gate=F.relu),
		critic_body=OneLayerFCBodyWithAction(
			phi_body.feature_dim, config.action_dim, 50, gate=F.relu),
		actor_opt_fn=lambda params: torch.optim.Adam(params, lr=3e-4),
		critic_opt_fn=lambda params: torch.optim.Adam(params, lr=3e-4))
	config.replay_fn = lambda: Replay(memory_size=int(1e4), batch_size=32)
	config.discount = 0.95
	config.state_normalizer = ImageNormalizer()
	config.max_steps = 1e7
	config.random_process_fn = lambda: OrnsteinUhlenbeckProcess(
		size=(config.action_dim, ), std=LinearSchedule(10,0,3.5e5))
	config.min_memory_size = 64
	config.target_network_mix = 1e-3
	config.log_interval = 301
	config.max_eps = 10150
	config.logger = get_logger()
	config.save_interval = 3001
	config.tag = "getcupreward"
	my_agent = DDPGAgent(config)
	try:
		my_agent.load('data/model-%s-%s-%s.bin' % (config.name, config.task_name, config.tag))
	except Exception as e:
		print(e)
	run_steps(my_agent)

def ppo_continuous(task_name, mstep):
	config = Config()
	config.num_workers = 1
	config.name = "ppo"
	config.tag = "hugerollout"
	config.state_dim = 84
	config.action_dim = 7
	phi_body = NatureConvBody()
	config.state_normalizer = ImageNormalizer()
	config.task_fn = lambda : TaskMap[task_name](frame_skip=1, max_steps=mstep)
	config.task_name = config.task_fn().name
	config.eval_env = config.task_fn()
	config.network_fn = lambda: GaussianActorCriticNet(
		config.state_dim, config.action_dim, phi_body=phi_body, actor_body=FCBody(phi_body.feature_dim),
		critic_body=FCBody(phi_body.feature_dim))
	config.optimizer_fn = lambda params: torch.optim.Adam(params, 3e-4, eps=1e-5)
	config.discount = 0.95
	config.use_gae = True
	config.gae_tau = 0.95
	# config.entropy_weight = 0.5
	config.gradient_clip = 5
	config.rollout_length = 9000
	config.optimization_epochs = 5
	config.num_mini_batches = 900
	config.ppo_ratio_clip = 0.2
	config.log_interval = 300
	config.max_eps = 15000
	config.max_steps = 4e5
	config.save_interval = 3000
	config.logger = get_logger()
	my_agent = PPOAgent(config)
	try:
		my_agent.load('data/model-%s-%s-%s.bin' % (config.name, config.task_name, config.tag))
	except Exception as e:
		print(e)
	run_steps(my_agent)

def a2c_continuous(task_name, mstep):
	config = Config()
	config.name = "a2c"
	config.tag = "hugerollout"
	config.state_dim = 84
	config.action_dim = 7
	phi_body = NatureConvBody()
	config.history_length = 1
	config.num_workers = 1
	config.state_normalizer = ImageNormalizer()
	config.task_fn = lambda : TaskMap[task_name](frame_skip=1, max_steps=mstep)
	config.task_name = config.task_fn().name
	config.eval_env = config.task_fn()
	config.optimizer_fn = lambda params: torch.optim.RMSprop(params, lr=3e-4)
	config.network_fn = lambda: GaussianActorCriticNet(
		config.state_dim, config.action_dim, phi_body=phi_body, actor_body=FCBody(phi_body.feature_dim),
		critic_body=FCBody(phi_body.feature_dim))
	config.discount = 0.95
	config.use_gae = True
	config.gae_tau = 0.95
	config.entropy_weight = 0.01
	config.rollout_length = 9000
	config.gradient_clip = 5
	config.log_interval = 300
	config.save_interval = 3000
	config.max_eps = 15000
	config.max_steps = 4e5
	config.logger = get_logger()
	my_agent = A2CAgent(config)
	try:
		my_agent.load('data/model-%s-%s-%s.bin' % (config.name, config.task_name, config.tag))
	except Exception as e:
		print(e)
	run_steps(my_agent)

def run_rl():
	task_name = "CutCarrot"
	max_steps = 50
	ddpg_pixel(task_name, max_steps)

if __name__ == "__main__":
	run_rl()