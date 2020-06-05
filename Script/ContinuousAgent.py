import sys, math, copy, time, subprocess, os
import rospy, tf
import numpy as np
from io import BytesIO
from rapidjson import loads,dumps 
from socketClient import Client
from socketServer import Server
from PIL import Image
import logging

from agent import *
from component import *
from utils import *
from component.bench import Monitor
from skimage import color, transform

agent_init_state = {"Name":"Agent1", \
			"Actor":{"Loc":{"X":-720.0,"Y":210.0,"Z":40.0},\
				"Rot":{"Pitch":0.0,"Yaw":0.0,"Roll":0.0}}, \
			"LeftHand":{"Loc":{"X":60,"Y":-40,"Z":100},\
				"Rot":{"Pitch":0.0,"Yaw":0.0,"Roll":0.0},\
				"Thumb":{
					"Prox":{"Pitch":0.0,"Yaw":0.0,"Roll":0.0}, 
					"Inter":{"Pitch":0.0,"Yaw":0.0,"Roll":0.0}, 
					"Dist":{"Pitch":0.0,"Yaw":0.0,"Roll":0.0}},
				"Index":{
					"Prox":{"Pitch":0.0,"Yaw":0.0,"Roll":0.0}, 
					"Inter":{"Pitch":0.0,"Yaw":0.0,"Roll":0.0}, 
					"Dist":{"Pitch":0.0,"Yaw":0.0,"Roll":0.0}},
				"Middle":{
					"Prox":{"Pitch":0.0,"Yaw":0.0,"Roll":0.0}, 
					"Inter":{"Pitch":0.0,"Yaw":0.0,"Roll":0.0}, 
					"Dist":{"Pitch":0.0,"Yaw":0.0,"Roll":0.0}},
				"Ring":{
					"Prox":{"Pitch":0.0,"Yaw":0.0,"Roll":0.0}, 
					"Inter":{"Pitch":0.0,"Yaw":0.0,"Roll":0.0}, 
					"Dist":{"Pitch":0.0,"Yaw":0.0,"Roll":0.0}},
				"Pinky":{
					"Prox":{"Pitch":0.0,"Yaw":0.0,"Roll":0.0}, 
					"Inter":{"Pitch":0.0,"Yaw":0.0,"Roll":0.0}, 
					"Dist":{"Pitch":0.0,"Yaw":0.0,"Roll":0.0}}
				}


			"RightHand":{"Loc":{"X":60,"Y":40,"Z":100},\
				"Rot":{"Pitch":0.0,"Yaw":0.0,"Roll":0.0},\
				"Thumb":{
					"Prox":{"Pitch":0.0,"Yaw":0.0,"Roll":0.0}, 
					"Inter":{"Pitch":0.0,"Yaw":0.0,"Roll":0.0}, 
					"Dist":{"Pitch":0.0,"Yaw":0.0,"Roll":0.0}},
				"Index":{
					"Prox":{"Pitch":0.0,"Yaw":0.0,"Roll":0.0}, 
					"Inter":{"Pitch":0.0,"Yaw":0.0,"Roll":0.0}, 
					"Dist":{"Pitch":0.0,"Yaw":0.0,"Roll":0.0}},
				"Middle":{
					"Prox":{"Pitch":0.0,"Yaw":0.0,"Roll":0.0}, 
					"Inter":{"Pitch":0.0,"Yaw":0.0,"Roll":0.0}, 
					"Dist":{"Pitch":0.0,"Yaw":0.0,"Roll":0.0}},
				"Ring":{
					"Prox":{"Pitch":0.0,"Yaw":0.0,"Roll":0.0}, 
					"Inter":{"Pitch":0.0,"Yaw":0.0,"Roll":0.0}, 
					"Dist":{"Pitch":0.0,"Yaw":0.0,"Roll":0.0}},
				"Pinky":{
					"Prox":{"Pitch":0.0,"Yaw":0.0,"Roll":0.0}, 
					"Inter":{"Pitch":0.0,"Yaw":0.0,"Roll":0.0}, 
					"Dist":{"Pitch":0.0,"Yaw":0.0,"Roll":0.0}}
				}

			"Head": {"Rot":{"Pitch":-60.0,"Yaw":0.0,"Roll":0.0}}
			}

class ContinuousAgent(object):
	def __init__(
			self, init_state,\
			client_endpoint=("127.0.0.1", 10120), \
			server_endpoint=("127.0.0.1", 10121)):
		self.client_endpoint = client_endpoint
		self.server_endpoint = server_endpoint
		self.client = Client(self.client_endpoint)
		self.server = Server(self.server_endpoint)
		self.retry_num = 50
		self.FNULL = open(os.devnull, 'w')
		self.frame = 0
		self.epochs = 0
		self.restart_ue = 100
		self.init_state = copy.deepcopy(init_state)
		self.execCmd = "/home/binroot/VRTasks/OpeningBottle/LinuxNoEditor/VRInteractPlatform/Binaries/Linux/VRInteractPlatform"
		self.p = subprocess.Popen("exec "+self.execCmd, shell=True,
			stdin=None, stdout=self.FNULL, stderr=None, close_fds=True)

	def start(self):
		self.state = copy.deepcopy(self.init_state)
		self.frame = 1
		self.server.listen()
		self.client.connect()
		data_frame = self.send_tf()
		i = 0
		while data_frame == None and i < self.retry_num:
			print("sending tf again")
			data_frame = self.send_tf()
			i += 1

		if data_frame == None:
			self.reset()

		return data_frame

	def reset(self):
		if self.epochs % self.restart_ue == 0:
			if self.p:
				self.p.kill()
			self.p = subprocess.Popen("exec "+self.execCmd, shell=True,
			stdin=None, stdout=self.FNULL, stderr=None, close_fds=True)
		else:
			self.send_tf(reset=True)

		# self.send_tf(reset=True)
		self.stop()
		self.server = Server(self.server_endpoint)
		self.client = Client(self.client_endpoint)	
		self.epochs += 1
		return self.start()	

	def Move(self, entity, idx, delta=10, scale=1):
		move_dict = {0:"X", 1:"Y", 2:"Z"}
		axis = move_dict[idx]
		self.state[entity]["Loc"][axis] += scale*delta 

	def Rotate(self, entity, idx, delta=90, scale=1):
		rot = copy.deepcopy(self.state[entity]["Rot"])

		if entity == "Actor":
			assert(idx == 1)
		if entity == "Head":
			assert(idx == 0)

		rotate_dict = {0:"Pitch", 1:"Yaw", 2:"Roll"}
		axis = rotate_dict[idx]
		rot[axis] += scale*delta
		if rot[axis] > 180:
			rot[axis] -= 360
		if rot[axis] < -180:
			rot[axis] += 360
		self.state[entity]["Rot"][axis] = rot[axis]

	def step(self, action, scale=1.0):
		

	def stop(self):
		self.server.stop()
		self.client.disconnect()
		# if self.p:
		# 	self.p.kill()
		# 	del self.p
		# 	self.p = None

		if self.server:
			del self.server
			self.server = None

		if self.client:
			del self.client
			self.client = None

	def send_tf(self, Name="Agent1", reset=False):
		data = {}
		data['AgentName'] = Name
		data['State'] = {}
		data['reset'] = reset
		if not reset:
			data['State']["ActorPose"] = \
				copy.deepcopy({
					"ActorLoc": self.state["Actor"]["Loc"],\
					"ActorRot": self.state["Actor"]["Rot"],\
				})

			data['State']['HeadPose'] = {}

			data['State']['HeadPose']["HeadWorldTransform"] = \
				copy.deepcopy({
					"Rot": self.state["Actor"]["Rot"],\
					"Trsl": self.state["Actor"]["Loc"],\
					"Scale": {"X":1.0,"Y":1.0,"Z":1.0}
				})

			data['State']['HeadPose']["HeadWorldTransform"]["Rot"]["Pitch"] = \
				copy.deepcopy(self.state["Head"]["Rot"]["Pitch"])

			data['State']['HeadPose']["HeadWorldTransform"]["Trsl"]["Z"] += 180

			data['State']["LeftHandPose"] = \
				copy.deepcopy({
					"LeftHandWorldPos": self.state["LeftHand"]['Loc'],
					"LeftHandWorldRot": self.state["LeftHand"]['Rot'],
					"Thumb": self.state["LeftHand"]["Thumb"],
					"Index": self.state["LeftHand"]["Index"],
					"Middle": self.state["LeftHand"]["Middle"],
					"Ring": self.state["LeftHand"]["Ring"],
					"Pinky": self.state["LeftHand"]["Pinky"]
				})

			data['State']["LeftHandPose"]["LeftHandWorldPos"]["Z"] += self.state["Actor"]["Loc"]["Z"]
			x = data['State']["LeftHandPose"]["LeftHandWorldPos"]["X"]
			y = data['State']["LeftHandPose"]["LeftHandWorldPos"]["Y"]
			abs_length = np.sqrt(x**2+y**2)

			theta = math.atan2(x, y)/math.pi*180
			theta -= data['State']["ActorPose"]["ActorRot"]["Yaw"]

			HandTheta = data['State']["LeftHandPose"]["LeftHandWorldRot"]["Yaw"]
			HandTheta += data['State']["ActorPose"]["ActorRot"]["Yaw"]
			
			if HandTheta > 180:
				HandTheta -= 360
			if HandTheta < -180:
				HandTheta += 360

			data['State']["LeftHandPose"]["LeftHandWorldRot"]["Yaw"] = HandTheta
			x_new = abs_length*math.sin(theta/180*math.pi)+data['State']["ActorPose"]["ActorLoc"]["X"]
			y_new = abs_length*math.cos(theta/180*math.pi)+data['State']["ActorPose"]["ActorLoc"]["Y"]
			data['State']["LeftHandPose"]["LeftHandWorldPos"]["X"] = x_new
			data['State']["LeftHandPose"]["LeftHandWorldPos"]["Y"] = y_new

			data['State']["RightHandPose"] = \
				copy.deepcopy({
					"RightHandWorldPos": self.state["RightHand"]['Loc'],
					"RightHandWorldRot": self.state["RightHand"]['Rot'],
					"Thumb": self.state["RightHand"]["Thumb"],
					"Index": self.state["RightHand"]["Index"],
					"Middle": self.state["RightHand"]["Middle"],
					"Ring": self.state["RightHand"]["Ring"],
					"Pinky": self.state["RightHand"]["Pinky"]
				})

			data['State']["RightHandPose"]["RightHandWorldPos"]["Z"] += self.state["Actor"]["Loc"]["Z"]
			x = data['State']["RightHandPose"]["RightHandWorldPos"]["X"]
			y = data['State']["RightHandPose"]["RightHandWorldPos"]["Y"]
			abs_length = np.sqrt(x**2+y**2)

			theta = math.atan2(x, y)/math.pi*180
			theta -= data['State']["ActorPose"]["ActorRot"]["Yaw"]

			HandTheta = data['State']["RightHandPose"]["RightHandWorldRot"]["Yaw"]
			HandTheta += data['State']["ActorPose"]["ActorRot"]["Yaw"]
			
			if HandTheta > 180:
				HandTheta -= 360
			if HandTheta < -180:
				HandTheta += 360

			data['State']["RightHandPose"]["RightHandWorldRot"]["Yaw"] = HandTheta
			x_new = abs_length*math.sin(theta/180*math.pi)+data['State']["ActorPose"]["ActorLoc"]["X"]
			y_new = abs_length*math.cos(theta/180*math.pi)+data['State']["ActorPose"]["ActorLoc"]["Y"]
			data['State']["RightHandPose"]["RightHandWorldPos"]["X"] = x_new
			data['State']["RightHandPose"]["RightHandWorldPos"]["Y"] = y_new
			

		msg = dumps(data)
		msg = msg+"\n"
		self.client.send(msg.encode())
		if not reset:
			return self.receive_msg()

	def receive_msg(self):
		try:
			data_frame = {}
			depth_head = self.server.getBuffer()
			# print(depth_head)
			assert(depth_head == "Depth")
			depth_data = np.load(BytesIO(self.server.getBuffer()))
			
			rgb_head = self.server.getBuffer()
			assert(rgb_head == "RGB")
			rgb_data = np.load(BytesIO(self.server.getBuffer()))
			# rgb_image = Image.fromarray(rgb_data)
			# rgb_image.show()
			# rgb_image.close()

			reward_head = self.server.getBuffer()
			assert(reward_head == "Reward")
			reward = float(self.server.getBuffer())

			done_head = self.server.getBuffer()
			assert(done_head == "Done")
			done = (self.server.getBuffer() == "1")

			data_frame["frame"] = self.frame
			data_frame["depth"] = depth_data
			data_frame["rgb"] = rgb_data
			data_frame["reward"] = reward
			data_frame["done"] = done
			return data_frame

		except Exception as e:
			print e
			return None