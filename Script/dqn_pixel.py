##########################################
# Stat232A&CS266A Project 3:
# Solving CartPole with Deep Q-Network
# Author: Feng Gao
##########################################

import argparse
import gym
import random
import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import deque
from itertools import count
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as T


parser = argparse.ArgumentParser(description='DQN_AGENT')
parser.add_argument('--epochs', type=int, default=200, metavar='E',
					help='number of epochs to train (default: 300)')
parser.add_argument('--batch-size', type=int, default=64, metavar='B',
					help='batch size for training (default: 32)')
parser.add_argument('--memory-size', type=int, default=500, metavar='M',
					help='memory length (default: 1000)')
parser.add_argument('--max-step', type=int, default=250,
					help='max steps allowed in gym (default: 250)')

args = parser.parse_args()
args.cuda = torch.cuda.is_available()

resize = T.Compose([T.ToPILImage(),
					T.Resize(40, interpolation=Image.CUBIC),
					T.ToTensor()])

use_cuda = False
# torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor

class DQN(nn.Module):
	def __init__(self, num_classes=14):
		super(DQN, self).__init__()
###################################################################
# Image input network architecture and forward propagation. Dimension
# of output layer should match the number of actions.
###################################################################
		# Define your network structure here:
		self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
		self.bn1 = nn.BatchNorm2d(16)
		self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
		self.bn2 = nn.BatchNorm2d(32)
		self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
		self.bn3 = nn.BatchNorm2d(32)
		self.head = nn.Linear(256, num_classes)
		# Define your forward propagation function here:
	def forward(self, x):
		x = F.relu(self.bn1(self.conv1(x)))
		x = F.relu(self.bn2(self.conv2(x)))
		x = F.relu(self.bn3(self.conv3(x)))
		return self.head(x.view(x.size(0), -1))

	# 	self.features = nn.Sequential(
	# 		nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
	# 		nn.ReLU(inplace=True),
	# 		nn.MaxPool2d(kernel_size=3, stride=2),
	# 		nn.Conv2d(64, 192, kernel_size=5, padding=2),
	# 		nn.ReLU(inplace=True),
	# 		nn.MaxPool2d(kernel_size=3, stride=2),
	# 		nn.Conv2d(192, 384, kernel_size=3, padding=1),
	# 		nn.ReLU(inplace=True),
	# 		nn.Conv2d(384, 256, kernel_size=3, padding=1),
	# 		nn.ReLU(inplace=True),
	# 		nn.Conv2d(256, 256, kernel_size=3, padding=1),
	# 		nn.ReLU(inplace=True),
	# 		nn.MaxPool2d(kernel_size=3, stride=2),
	# 	)
	# 	self.classifier = nn.Sequential(
	# 		nn.Dropout(),
	# 		nn.Linear(256 * 6 * 6, 4096),
	# 		nn.ReLU(inplace=True),
	# 		nn.Dropout(),
	# 		nn.Linear(4096, 4096),
	# 		nn.ReLU(inplace=True),
	# 		nn.Linear(4096, num_classes),
	# 	)

	# def forward(self, x):
	# 	x = self.features(x)
	# 	x = x.view(x.size(0), 256 * 6 * 6)
	# 	x = self.classifier(x)
	# 	return x

###################################################################
# State vector input network architecture and forward propagation.
# Dimension of output layer should match the number of actions.
##################################################################
	# 	# Define your network structure here (no need to have conv
	# 	# block for state input):
	# 	self.fc1 = nn.Linear(4, 256)
	# 	self.fc2 = nn.Linear(256, 2)
	# 	# Define your forward propagation function here:	
	# def forward(self, x):
	# 	x = F.relu(self.fc1(x))
	# 	return self.fc2(x)

class DQNagent():
	def __init__(self):
		self.model = DQN().cuda() if use_cuda else DQN()
		self.memory = deque(maxlen=args.memory_size)
		self.gamma = 0.9999
		self.epsilon_start = 1
		self.epsilon_min = 0.01
		self.epsilon_decay = 1000
		self.steps = 0
###################################################################
# remember() function
# remember function is for the agent to get "experience". Such experience
# should be storaged in agent's memory. The memory will be used to train
# the network. The training example is the transition: (state, action,
# next_state, reward). There is no return in this function, instead,
# you need to keep pushing transition into agent's memory. For your
# convenience, agent's memory buffer is defined as deque.
###################################################################
	def remember(self, state, action, next_state, reward):
		expr = (state, action, next_state, reward)
		self.memory.append(expr)
###################################################################
# act() fucntion
# This function is for the agent to act on environment while training.
# You need to integrate epsilon-greedy in it. Please note that as training
# goes on, epsilon should decay but not equal to zero. We recommend to
# use the following decay function:
# epsilon = epsilon_min+(epsilon_start-epsilon_min)*exp(-1*global_step/epsilon_decay)
# act() function should return an action according to epsilon greedy. 
# Action is index of largest Q-value with probability (1-epsilon) and 
# random number in [0,1] with probability epsilon.
###################################################################
	def act(self, state):
		epsilon = self.epsilon_min+(self.epsilon_start-self.epsilon_min) \
			*np.exp(-1.*self.steps/self.epsilon_decay)
		self.steps += 1
		if random.uniform(0, 1) < epsilon:
			return random.randint(0, 13)
		else:
			qvalues = self.model(Variable(state, volatile = True).type(FloatTensor))
			qvalues = qvalues.cpu().data.numpy() if use_cuda else qvalues.data.numpy()
			# print(qvalues)
			return np.argmax(qvalues)


###################################################################
# replay() function
# This function performs an one step replay optimization. It first
# samples a batch from agent's memory. Then it feeds the batch into 
# the network. After that, you will need to implement Q-Learning. 
# The target Q-value of Q-Learning is Q(s,a) = r + gamma*max_{a'}Q(s',a'). 
# The loss function is distance between target Q-value and current
# Q-value. We recommend to use F.smooth_l1_loss to define the distance.
# There is no return of act() function.
# Please be noted that parameters in Q(s', a') should not be updated.
# You may use Variable().detach() to detach Q-values of next state 
# from the current graph.
###################################################################
	def replay(self, batch_size, optimizer):
		if len(self.memory) < batch_size:
			return 1
		batch = random.sample(self.memory, batch_size)
		state_batch = Tensor()
		action_batch = Tensor(batch_size).zero_()
		next_state_batch = Tensor()
		reward_batch = Tensor(batch_size).zero_()
		non_final_state_mask = ByteTensor(batch_size).zero_()

		for i in range(batch_size):
			# print((state_batch, batch[i][0].float()))
			state_batch = torch.cat((state_batch, batch[i][0].float()))
			action_batch[i] = batch[i][1]
			if not batch[i][2] is None:
				next_state_tensor = batch[i][2].float().cuda() if use_cuda else batch[i][2].float()
				next_state_batch = torch.cat((next_state_batch, next_state_tensor))
				non_final_state_mask[i] = True
			reward_batch[i] = batch[i][3]

		state_batch = Variable(state_batch)
		action_batch = Variable(action_batch).type(LongTensor).view(-1,1)
		next_state_batch = Variable(next_state_batch, volatile=True)
		reward_batch = Variable(reward_batch)
		orig_qvalues = self.model(state_batch)
		orig_qvalues = orig_qvalues.gather(1, action_batch)
		next_qvalues = Variable(Tensor(batch_size).zero_())
		next_qvalues[non_final_state_mask] = self.model(next_state_batch).max(1)[0]
		next_qvalues.volatile = False

		targ_qvalues = reward_batch + self.gamma * next_qvalues
		loss = F.smooth_l1_loss(orig_qvalues, targ_qvalues)
		optimizer.zero_grad()
		loss.backward()
		for param in self.model.parameters():
			param.grad.data.clamp_(-1, 1)
		optimizer.step()
		return np.float(loss.cpu().data.numpy() if use_cuda else loss.data.numpy())


#################################################################
# Functions 'getCartLocation' and 'getGymScreen' are designed for 
# capturing current renderred image in gym. You can directly take 
# the return of 'getGymScreen' function, which is a resized image
# with size of 3*40*80.
#################################################################

def getCartLocation():
	world_width = env.x_threshold*2
	scale = 600/world_width
	return int(env.state[0]*scale+600/2.0)

def getGymScreen():
	screen = env.render(mode='rgb_array').transpose((2,0,1))
	screen = screen[:, 160:320]
	view_width = 320
	cart_location = getCartLocation()
	if cart_location < view_width//2:
		slice_range = slice(view_width)
	elif cart_location > (600-view_width//2):
		slice_range = slice(-view_width, None)
	else:
		slice_range = slice(cart_location - view_width//2, cart_location+view_width//2)
	screen = screen[:, :, slice_range]
	screen = np.ascontiguousarray(screen, dtype=np.float32)/255
	screen = FloatTensor(screen)
	return resize(screen).unsqueeze(0)

def plot_durations(durations):
	plt.figure(2)
	plt.clf()
	durations_t = torch.FloatTensor(durations)
	plt.title('Training...')
	plt.xlabel('Episode')
	plt.ylabel('Mean reward')
	plt.plot(durations_t.numpy())
	# Take 100 episode averages and plot them too
	mean_start = 30
	if len(durations_t) >= mean_start:
		means = durations_t.unfold(0, mean_start, 1).mean(1).view(-1)
		means = torch.cat((torch.zeros(mean_start-1), means))
		plt.plot(means.numpy())
	plt.pause(0.001)  # pause a bit so that plots are updated
	# print(len(durations_t))
	# if len(durations_t) == 199:
		# plt.savefig('testplot.jpg')
			

def plot_losses(losses):
	plt.figure(3)
	plt.clf()
	losses_t = torch.FloatTensor(losses)
	plt.title('Training...')
	plt.xlabel('Episode')
	plt.ylabel('Loss')
	plt.plot(losses_t.numpy())
	# Take 100 episode averages and plot them too
	if len(losses_t) >= 50:
		means = losses_t.unfold(0, 100, 1).mean(1).view(-1)
		means = torch.cat((torch.zeros(99), means))
		plt.plot(means.numpy())

	plt.pause(0.001)  # pause a bit so that plots are updated


# def main():
# 	global env, optimizer
# 	env = gym.make('CartPole-v0').unwrapped
# 	env._max_episode_steps = args.max_step
# 	print('env max steps:{}'.format(env._max_episode_steps))
# 	steps_done = 0
# 	agent = DQNagent()
# 	optimizer = optim.RMSprop(filter(lambda p: p.requires_grad, agent.model.parameters()), lr=1e-3)
# 	durations = []
# 	losses = []
# 	################################################################
# 	# training loop
# 	# You need to implement the training loop here. In each epoch, 
# 	# play the game until trial ends. At each step in one epoch, agent
# 	# need to remember the transitions in self.memory and perform
# 	# one step replay optimization. Use the following function to 
# 	# interact with the environment:
# 	#   env.step(action)
# 	# It gives you infomation about next step after taking the action.
# 	# The return of env.step() is (next_state, reward, done, info). You
# 	# do not need to use 'info'. 'done=1' means current trial ends.
# 	# if done equals to 1, please use -1 to substitute the value of reward.
# 	################################################################
# 	for epoch in range(args.epochs):
# 		steps = 0
# 		done = False
# 	################################################################
# 	# Image input. We recommend to use the difference between two
# 	# images of current_screen and last_screen as input image.
# 	################################################################
# 		env.reset()
# 		last_screen = getGymScreen()
# 		current_screen = getGymScreen()
# 		state = current_screen - last_screen
# 		for steps in range(args.max_step):
# 			env.render()
# 			action = agent.act(state)
# 			_, reward, done, _ = env.step(action)
# 			last_screen = current_screen
# 			current_screen = getGymScreen()
# 			if not done:
# 				next_state = current_screen - last_screen
# 			else:
# 				next_state = None
# 				reward = -20
# 			agent.remember(state, action, next_state, reward)
# 			state = next_state
# 			loss = agent.replay(64)
# 			if done or steps == args.max_step-1:
# 				print("Episode finished after {} timesteps".format(steps))
# 				durations.append(steps)
# 				# losses.append(loss)
# 				plot_durations(durations)
# 				# plot_losses(losses)
# 				break
	################################################################
	# State vector input. You can direct take observation from gym 
	# as input of agent's DQN
	################################################################
		# state = torch.from_numpy(env.reset()).view(1,4).float()
		# for steps in range(args.max_step):
		# 	env.render()
		# 	# print(state)
		# 	action = agent.act(state)
		# 	next_state, reward, done, _ = env.step(action)
		# 	next_state = torch.from_numpy(next_state).view(1,4).float()
		# 	if done:
		# 		reward = -20
		# 		next_state = None
		# 	agent.remember(state, action, next_state, reward)
		# 	loss = agent.replay(32)
		# 	state = next_state
		# 	if done or steps == args.max_step-1:
		# 		print("Episode finished after {} timesteps".format(steps))
		# 		durations.append(steps)
		# 		# losses.append(loss)
		# 		plot_durations(durations)
		# 		# plot_losses(losses)
		# 		break


	################################################################

# if __name__ == "__main__":
# 	main()
