import numpy as np
from deep_rl import *
from api_dish import GoTo, Take, PlaceTo, Use, Open, env, ObjDict, InitDict
from skimage import color, transform
from torchvision import transforms
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
import torch.optim as optim
import random
import time
from PIL import Image

class Cut():
	def __init__(self, name="Cut", sub_goal=[None], goal_states=[None], max_steps=10000, frame_size=84, frame_skip=1):
		self.steps = 0
		self.max_steps = max_steps
		self.frame_size = frame_size
		self.frame_skip = frame_skip
		self.env = None
		self.state_dim = frame_size
		self.action_dim = 7
		self.name = name
		self.goal_states = goal_states[1:]
		self.sub_goal = sub_goal

	def start(self):
		if self.env == None:
			self.env = env
		self.env.start()

	def reset(self):
		self.steps = 0
		if self.env == None:
			self.env = env
			data = self.env.start()
		else:
			# print("reset")
			data = self.env.reset()
		state = torch.from_numpy(np.expand_dims(data['rgb'],axis=0).transpose((0, 3, 1, 2)))
		state = state.float().cuda()
		del data
		InitDict()

		return state

	def normalize_state(self, state):
		return np.asarray(state) / 255.0

	# rule based discriminator
	def disc(self, sg, state):
		all_legal_states = []

		for i in range(self.sub_goal.shape[0]):
			if not (self.sub_goal[i]-sg).cpu().byte().any():
				all_legal_states.append(self.goal_states[i])

		# all_states = [self.goal_states[i] for i, x in enumerate(self.sub_goal) if x == sg]
		# print(all_legal_states)
		# print(state)
		# print("all goal states", len(all_states))
		for s in all_legal_states:
			if s == state:
				reward = 1.0
				done = True
				break
			else:
				reward = -0.01
				done = False

		return reward, done


	def step(self, a, sub_goal=[], rollout=False):
		action = a
		# print("action", a)
		if action == 0:
			data = GoTo("Fridge")
		elif action == 1:
			data = GoTo("Knife")
		elif action == 2:
			data = Open("Fridge")
		elif action == 3:
			data = Use("Knife")
		elif action == 4:
			data = Take("Tomato")
		elif action == 5:
			env.state['rgb'] = True
			data = env.step("ActorRotateRight")
			env.state['rgb'] = False
		elif action == 6:
			env.state['rgb'] = True
			data = env.step("ActorRotateLeft")
			env.state['rgb'] = False
		else:
			print("action out of bound!")
			return None

		# print(data)
		next_states = torch.from_numpy(np.expand_dims(data['rgb'],axis=0).transpose((0, 3, 1, 2)))
		next_states = next_states.float().cuda()

		if len(sub_goal) > 0:
			reward, done = self.disc(sub_goal, str(ObjDict))
		else:
			reward = data['reward']
			done = data['done']

		del data
		info = None
		self.steps += 1
		done = (done or self.steps >= self.max_steps)

		if done and not rollout:
			self.reset()

		if reward == 0:
			reward = -0.01
			
		return next_states, reward, done, info

def test():
	task_list = [5,0,2,4,6,1,3]
	task_fn = lambda : Cut(frame_skip=1, max_steps=100)
	task = task_fn()
	task.start()
	# task.reset()
	for a in task_list:
		a,b,c,d = task.step(a)
		print("reward", b)

if __name__ == "__main__":
	test()