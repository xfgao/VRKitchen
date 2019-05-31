import numpy as np
from DiscreteAgent import DiscreteAgent
from skimage import color, transform
import torch
import random

# torch.manual_seed(0)
# np.random.seed(0)
# random.seed(0)

class CutCarrot():
	def __init__(self, name="CutCarrot", max_steps=30, frame_size=84, frame_skip=1):
		self.steps = 0
		self.max_steps = max_steps
		self.frame_size = frame_size
		self.frame_skip = frame_skip
		self.env = None
		self.state_dim = frame_size
		self.action_dim = 7
		self.name = name
		self.init_state = {"Name":"Agent1", \
			"Actor":{"Loc":{"X":-735.0,"Y":297.0,"Z":35.0},\
				"Rot":{"Pitch":0.0,"Yaw":0.0,"Roll":0.0}}, \

			"LeftHand":{"Grab": False, "Release": False, "ActorName":"", "CompName":"",
				"NeutralLoc":{"X":40,"Y":-10,"Z":120},"NeutralRot":{"Pitch":0.0,"Yaw":0.0,"Roll":0.0},
				"Loc":{"X":40,"Y":-10,"Z":120},"Rot":{"Pitch":0.0,"Yaw":0.0,"Roll":0.0},
				"WorldLoc":{"X":0,"Y":0,"Z":0}},\

			"RightHand":{"Grab": False, "Release": False, "ActorName":"", "CompName":"",\
				"NeutralLoc":{"X":40,"Y":10,"Z":120}, "NeutralRot":{"Pitch":0.0,"Yaw":0.0,"Roll":0.0}, \
				"Loc":{"X":40,"Y":10,"Z":120}, "Rot":{"Pitch":0.0,"Yaw":0.0,"Roll":0.0}, \
				"WorldLoc":{"X":0,"Y":0,"Z":0}},

			"Head": {"Rot":{"Pitch":-45,"Yaw":0.0,"Roll":0.0}},
			"rgb": True, "depth": False, "mask": False
			}

	def start(self):
		self.env.start()

	def reset(self):
		self.steps = 0
		if self.env == None:
			self.env = DiscreteAgent(self.init_state)
			data = self.env.start()
		else:
			print("reset")
			data = self.env.reset()
		
		state = torch.from_numpy(np.moveaxis(data['rgb'], -1, 0))
		state = state.float()
		del data

		return state

	def normalize_state(self, state):
		return np.asarray(state) / 255.0

	def step(self, a):
		action = "ControlRightHand"
		a.resize((7))
		data = self.env.step(action, scale=30, loc=a[0:3], rot=a[3:6], grab_strength=a[6], grab_actor="Knife", grab_comp="StaticMeshComponent0")
		next_states = torch.from_numpy(np.moveaxis(data['rgb'], -1, 0))
		next_states = next_states.float()

		reward = data['reward']
		done = data['done']
		del data
		info = None
		self.steps += 1
		done = (done or self.steps >= self.max_steps)

		if done:
			next_states = self.reset()
		return next_states, reward, done, info

class PourWater():
	def __init__(self, name="PourWater", max_steps=30, frame_size=84, frame_skip=1):
		self.steps = 0
		self.max_steps = max_steps
		self.frame_size = frame_size
		self.frame_skip = frame_skip
		self.env = None
		self.name = name
		self.state_dim = frame_size
		self.action_dim = 7
		self.init_state = {"Name":"Agent1", \
			"Actor":{"Loc":{"X":-735.0,"Y":80.0,"Z":35.0},\
				"Rot":{"Pitch":0.0,"Yaw":0.0,"Roll":0.0}}, \

			"LeftHand":{"Grab": False, "Release": False, "ActorName":"", "CompName":"",
				"NeutralLoc":{"X":40,"Y":-10,"Z":120},"NeutralRot":{"Pitch":0.0,"Yaw":0.0,"Roll":0.0},
				"Loc":{"X":40,"Y":-10,"Z":120},"Rot":{"Pitch":0.0,"Yaw":0.0,"Roll":0.0},
				"WorldLoc":{"X":0,"Y":0,"Z":0}},\

			"RightHand":{"Grab": False, "Release": False, "ActorName":"", "CompName":"",\
				"NeutralLoc":{"X":40,"Y":10,"Z":120}, "NeutralRot":{"Pitch":0.0,"Yaw":0.0,"Roll":0.0}, \
				"Loc":{"X":40,"Y":10,"Z":120}, "Rot":{"Pitch":0.0,"Yaw":0.0,"Roll":0.0}, \
				"WorldLoc":{"X":0,"Y":0,"Z":0}},

			"Head": {"Rot":{"Pitch":-45,"Yaw":0.0,"Roll":0.0}},
			"rgb": True, "depth": False, "mask": False

			}

	def start(self):
		self.env.start()

	def reset(self):
		self.steps = 0
		if self.env == None:
			self.env = DiscreteAgent(self.init_state)
			data = self.env.start()
		else:
			print("reset")
			data = self.env.reset()
		
		state = torch.from_numpy(np.moveaxis(data['rgb'], -1, 0))
		state = state.float()
		del data

		return state

	def normalize_state(self, state):
		return np.asarray(state) / 255.0

	def step(self, a):
		action = "ControlRightHand"
		a.resize((7))
		data = self.env.step(action, scale=10, loc=a[0:3], rot=a[3:6], grab_strength=a[6], grab_actor="Cup2", grab_comp="ContainerMesh")
		next_states = torch.from_numpy(np.moveaxis(data['rgb'], -1, 0))
		next_states = next_states.float()

		reward = data['reward']
		done = data['done']
		del data
		info = None
		self.steps += 1
		done = (done or self.steps >= self.max_steps)

		if done:
			next_states = self.reset()
		return next_states, reward, done, info

class GetWater():
	def __init__(self, name="GetWater", max_steps=30, frame_size=84, frame_skip=1):
		self.steps = 0
		self.max_steps = max_steps
		self.frame_size = frame_size
		self.frame_skip = frame_skip
		self.env = None
		self.name = name
		self.state_dim = frame_size
		self.action_dim = 7
		self.init_state = {"Name":"Agent1", \
			"Actor":{"Loc":{"X":-1071.0,"Y":164.0,"Z":35.0},\
				"Rot":{"Pitch":0.0,"Yaw":180.0,"Roll":0.0}}, \

			"LeftHand":{"Grab": False, "Release": False, "ActorName":"", "CompName":"",
				"NeutralLoc":{"X":40,"Y":-10,"Z":120},"NeutralRot":{"Pitch":0.0,"Yaw":0.0,"Roll":0.0},
				"Loc":{"X":40,"Y":-10,"Z":120},"Rot":{"Pitch":0.0,"Yaw":0.0,"Roll":0.0},
				"WorldLoc":{"X":0,"Y":0,"Z":0}},\

			"RightHand":{"Grab": False, "Release": False, "ActorName":"", "CompName":"",\
				"NeutralLoc":{"X":40,"Y":10,"Z":120}, "NeutralRot":{"Pitch":0.0,"Yaw":0.0,"Roll":0.0}, \
				"Loc":{"X":40,"Y":10,"Z":120}, "Rot":{"Pitch":0.0,"Yaw":0.0,"Roll":0.0}, \
				"WorldLoc":{"X":0,"Y":0,"Z":0}},

			"Head": {"Rot":{"Pitch":-45,"Yaw":0.0,"Roll":0.0}},
			"rgb": True, "depth": False, "mask": False

			}

	def start(self):
		self.env.start()

	def reset(self):
		self.steps = 0
		if self.env == None:
			self.env = DiscreteAgent(self.init_state)
			data = self.env.start()
		else:
			print("reset")
			data = self.env.reset()
		
		state = torch.from_numpy(np.moveaxis(data['rgb'], -1, 0))
		state = state.float()
		del data

		return state

	def normalize_state(self, state):
		return np.asarray(state) / 255.0

	def step(self, a):
		action = "ControlRightHand"
		a.resize((7))
		data = self.env.step(action, scale=30, loc=a[0:3], rot=a[3:6], grab_strength=a[6], grab_actor="Cup3", grab_comp="ContainerMesh")
		next_states = torch.from_numpy(np.moveaxis(data['rgb'], -1, 0))
		next_states = next_states.float()

		reward = data['reward']
		done = data['done']
		del data
		info = None
		self.steps += 1
		done = (done or self.steps >= self.max_steps)

		if done:
			next_states = self.reset()
		return next_states, reward, done, info

class OpenCan():
	def __init__(self, name="OpenCan", max_steps=30, frame_size=84, frame_skip=1):
		self.steps = 0
		self.max_steps = max_steps
		self.frame_size = frame_size
		self.frame_skip = frame_skip
		self.env = None
		self.name = name
		self.state_dim = frame_size
		self.action_dim = 7
		self.init_state = {"Name":"Agent1", \
			"Actor":{"Loc":{"X":-1100.0,"Y":365.0,"Z":35.0},\
				"Rot":{"Pitch":0.0,"Yaw":90.0,"Roll":0.0}}, \

			"LeftHand":{"Grab": False, "Release": False, "ActorName":"", "CompName":"",
				"NeutralLoc":{"X":40,"Y":-10,"Z":120},"NeutralRot":{"Pitch":0.0,"Yaw":0.0,"Roll":0.0},
				"Loc":{"X":40,"Y":-10,"Z":120},"Rot":{"Pitch":0.0,"Yaw":0.0,"Roll":0.0},
				"WorldLoc":{"X":0,"Y":0,"Z":0}},\

			"RightHand":{"Grab": False, "Release": False, "ActorName":"", "CompName":"",\
				"NeutralLoc":{"X":40,"Y":10,"Z":120}, "NeutralRot":{"Pitch":0.0,"Yaw":0.0,"Roll":0.0}, \
				"Loc":{"X":40,"Y":10,"Z":120}, "Rot":{"Pitch":0.0,"Yaw":0.0,"Roll":0.0}, \
				"WorldLoc":{"X":0,"Y":0,"Z":0}},

			"Head": {"Rot":{"Pitch":-45,"Yaw":0.0,"Roll":0.0}},
			"rgb": True, "depth": False, "mask": False

			}

	def start(self):
		self.env.start()

	def reset(self):
		self.steps = 0
		if self.env == None:
			self.env = DiscreteAgent(self.init_state)
			data = self.env.start()
		else:
			print("reset")
			data = self.env.reset()
		
		state = torch.from_numpy(np.moveaxis(data['rgb'], -1, 0))
		state = state.float()
		del data

		return state

	def normalize_state(self, state):
		return np.asarray(state) / 255.0

	def step(self, a):
		action = "ControlRightHand"
		a.resize((7))
		data = self.env.step(action, scale=30, loc=a[0:3], rot=a[3:6], grab_strength=a[6], grab_actor="CanOpener", grab_comp="StaticMeshComponent0")
		next_states = torch.from_numpy(np.moveaxis(data['rgb'], -1, 0))
		next_states = next_states.float()

		reward = data['reward']
		done = data['done']
		del data
		info = None
		self.steps += 1
		done = (done or self.steps >= self.max_steps)

		if done:
			next_states = self.reset()
		return next_states, reward, done, info

class PeelKiwi():
	def __init__(self, name="PeelKiwi", max_steps=30, frame_size=84, frame_skip=1):
		self.steps = 0
		self.max_steps = max_steps
		self.frame_size = frame_size
		self.frame_skip = frame_skip
		self.env = None
		self.name = name
		self.state_dim = frame_size
		self.action_dim = 7
		self.init_state = {"Name":"Agent1", \
			"Actor":{"Loc":{"X":-1070.0,"Y":-6.0,"Z":35.0},\
				"Rot":{"Pitch":0.0,"Yaw":180.0,"Roll":0.0}}, \

			"LeftHand":{"Grab": False, "Release": False, "ActorName":"", "CompName":"",
				"NeutralLoc":{"X":40,"Y":-10,"Z":120},"NeutralRot":{"Pitch":0.0,"Yaw":0.0,"Roll":0.0},
				"Loc":{"X":40,"Y":-10,"Z":120},"Rot":{"Pitch":0.0,"Yaw":0.0,"Roll":0.0},
				"WorldLoc":{"X":0,"Y":0,"Z":0}},\

			"RightHand":{"Grab": False, "Release": False, "ActorName":"", "CompName":"",\
				"NeutralLoc":{"X":40,"Y":10,"Z":120}, "NeutralRot":{"Pitch":0.0,"Yaw":0.0,"Roll":0.0}, \
				"Loc":{"X":40,"Y":10,"Z":120}, "Rot":{"Pitch":0.0,"Yaw":0.0,"Roll":0.0}, \
				"WorldLoc":{"X":0,"Y":0,"Z":0}},

			"Head": {"Rot":{"Pitch":-45,"Yaw":0.0,"Roll":0.0}},
			"rgb": True, "depth": False, "mask": False

			}

	def start(self):
		self.env.start()

	def reset(self):
		self.steps = 0
		if self.env == None:
			self.env = DiscreteAgent(self.init_state)
			data = self.env.start()
		else:
			print("reset")
			data = self.env.reset()
		
		state = torch.from_numpy(np.moveaxis(data['rgb'], -1, 0))
		state = state.float()
		del data

		return state

	def normalize_state(self, state):
		return np.asarray(state) / 255.0

	def step(self, a):
		action = "ControlRightHand"
		a.resize((7))
		data = self.env.step(action, scale=30, loc=a[0:3], rot=a[3:6], grab_strength=a[6], grab_actor="Peeler", grab_comp="StaticMeshComponent0")
		next_states = torch.from_numpy(np.moveaxis(data['rgb'], -1, 0))
		next_states = next_states.float()

		reward = data['reward']
		done = data['done']
		del data
		info = None
		self.steps += 1
		done = (done or self.steps >= self.max_steps)

		if done:
			next_states = self.reset()
		return next_states, reward, done, info
