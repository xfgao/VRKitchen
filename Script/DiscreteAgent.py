import sys, math, copy, time, subprocess, os
#import rospy, tf
import numpy as np
from io import BytesIO, StringIO
from rapidjson import loads,dumps 
from socketClient import Client
from socketServerU import Server
from PIL import Image
import logging
from tool_pos import tool_pos

from skimage import color, transform

# agent_init_state = {"Name":"Agent1", \
# 			"Actor":{"Loc":{"X":0.0,"Y":0.0,"Z":0.0},\
# 				"Rot":{"Pitch":0.0,"Yaw":0.0,"Roll":0.0}}, \

# 			"LeftHand":{"Grab": False, "Release": False, "ActorName":"", "CompName":"",
# 				"NeutralLoc":{"X":40,"Y":-10,"Z":120},"NeutralRot":{"Pitch":0.0,"Yaw":0.0,"Roll":0.0},
# 				"Loc":{"X":40,"Y":-10,"Z":120},"Rot":{"Pitch":0.0,"Yaw":0.0,"Roll":0.0},
# 				"WorldLoc":{"X":0,"Y":0,"Z":0}},\

# 			"RightHand":{"Grab": False, "Release": False, "ActorName":"", "CompName":"",\
# 				"NeutralLoc":{"X":40,"Y":10,"Z":120}, "NeutralRot":{"Pitch":0.0,"Yaw":0.0,"Roll":0.0}, \
# 				"Loc":{"X":40,"Y":10,"Z":120}, "Rot":{"Pitch":0.0,"Yaw":0.0,"Roll":0.0}, \
# 				"WorldLoc":{"X":0,"Y":0,"Z":0}},

# 			"Head": {"Rot":{"Pitch":0,"Yaw":0.0,"Roll":0.0}},
# 			"rgb": False, "depth": False, "mask": False
# 			}


SPEED = 20

class DiscreteAgent(object):
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
		self.data = {}
		self.restart_ue = 100
		self.init_state = copy.deepcopy(init_state)
		self.execCmd = "/home/binroot/UE4Packages/test1/LinuxNoEditor/VRInteractPlatform/Binaries/Linux/VRInteractPlatform"
		self.crouch = False
		# self.p = subprocess.Popen("exec "+self.execCmd, shell=True,\
		# 	stdin=None, stdout=self.FNULL, stderr=None, close_fds=True)

	def start(self, level=0):
		self.state = copy.deepcopy(self.init_state)
		self.state_old = copy.deepcopy(self.init_state)
		if level:
			self.state['scene'] = level
			self.state_old['scene'] = level
		self.frame = 1
		self.server.listen()
		self.client.connect()
		data_frame = self.send_tf()
		i = 0
		while data_frame == None and i < self.retry_num:
			print("sending tf again")
			data_frame = self.send_tf()
			i += 1

		self.data = data_frame

		if data_frame == None:
			self.reset(level)
		return data_frame

	def reset(self, level=0):
		# if self.epochs % self.restart_ue == 0:
		# 	if self.p:
		# 		self.p.kill()
		# 	self.p = subprocess.Popen("exec "+self.execCmd, shell=True,
		# 	stdin=None, stdout=self.FNULL, stderr=None, close_fds=True)
		# else:
		# 	self.send_tf(reset=True)

		if level:
			self.state['scene'] = level
			self.state_old['scene'] = level
		self.send_tf(reset=True)

		self.stop()
		self.server = Server(self.server_endpoint)
		self.client = Client(self.client_endpoint)	
		self.epochs += 1
		return self.start(level)

	def stop(self):
		self.server.stop()
		# self.client.disconnect()
		# if self.p:
		# 	self.p.kill()
		# 	del self.p
		# 	self.p = None

		# if self.server:
		# 	del self.server
		# 	self.server = None

		# if self.client:
		# 	del self.client
		# 	self.client = None

	def send_tf(self, Name="Agent1", reset=False, world=False, anim_speed=0.0):
		data = {}
		data['AgentName'] = Name
		data['State'] = {}
		data['reset'] = reset
		data['scene'] = self.state['scene']

		if not reset:
			data['State']["ActorPose"] = \
				copy.deepcopy({
					"ActorLoc": self.state["Actor"]["Loc"],\
					"ActorRot": self.state["Actor"]["Rot"],\
					"CurrentSpeed": anim_speed
				})

			data['State']['HeadPose'] = {}
			data['State']['crouch'] = {}
			data['State']['HeadPose']["HeadWorldTransform"] = \
				copy.deepcopy({
					"Rot": self.state["Actor"]["Rot"],\
					"Trsl": self.state["Actor"]["Loc"],\
					"Scale": {"X":1.0,"Y":1.0,"Z":1.0}
				})

			data['State']['HeadPose']["HeadWorldTransform"]["Rot"]["Pitch"] = \
				copy.deepcopy(self.state["Head"]["Rot"]["Pitch"])

			data['State']['HeadPose']["HeadWorldTransform"]["Trsl"]["Z"] += 180

			data['State']["LeftHandGrab"] = \
				copy.deepcopy({
					"LeftGrab": self.state["LeftHand"]['Grab'],
					"LeftRelease": self.state["LeftHand"]["Release"],
					"ActorName": self.state["LeftHand"]["ActorName"],
					"CompName": self.state["LeftHand"]["CompName"]
				})

			data['State']["LeftHandPose"] = \
				copy.deepcopy({
					"LeftHandWorldPos": self.state["LeftHand"]['Loc'],
					"LeftHandWorldRot": self.state["LeftHand"]['Rot']
				})

			data['State']["RightHandGrab"] = \
				copy.deepcopy({
					"RightGrab": self.state["RightHand"]['Grab'],
					"RightRelease": self.state["RightHand"]['Release'],
					"ActorName": self.state["RightHand"]["ActorName"],
					"CompName": self.state["RightHand"]["CompName"]
				})

			data['State']["RightHandPose"] = \
				copy.deepcopy({
					"RightHandWorldPos": self.state["RightHand"]['Loc'],
					"RightHandWorldRot": self.state["RightHand"]['Rot']
				})

			x = data['State']["LeftHandPose"]["LeftHandWorldPos"]["X"]
			y = data['State']["LeftHandPose"]["LeftHandWorldPos"]["Y"]
			z = data['State']["LeftHandPose"]["LeftHandWorldPos"]["Z"]
			HandTheta = data['State']["LeftHandPose"]["LeftHandWorldRot"]["Yaw"]
			x_new, y_new, z_new, hand_new = self.RelToWorld(x, y ,z, HandTheta)
			data['State']["LeftHandPose"]["LeftHandWorldRot"]["Yaw"] = hand_new
			data['State']["LeftHandPose"]["LeftHandWorldPos"]["X"] = x_new
			data['State']["LeftHandPose"]["LeftHandWorldPos"]["Y"] = y_new
			data['State']["LeftHandPose"]["LeftHandWorldPos"]["Z"] = z_new

			x = data['State']["RightHandPose"]["RightHandWorldPos"]["X"]
			y = data['State']["RightHandPose"]["RightHandWorldPos"]["Y"]
			z = data['State']["RightHandPose"]["RightHandWorldPos"]["Z"]
			HandTheta = data['State']["RightHandPose"]["RightHandWorldRot"]["Yaw"]
			x_new, y_new, z_new, hand_new = self.RelToWorld(x, y ,z, HandTheta)
			data['State']["RightHandPose"]["RightHandWorldRot"]["Yaw"] = hand_new
			data['State']["RightHandPose"]["RightHandWorldPos"]["X"] = x_new
			data['State']["RightHandPose"]["RightHandWorldPos"]["Y"] = y_new
			data['State']["RightHandPose"]["RightHandWorldPos"]["Z"] = z_new
			data['State']["crouch"]["crouch"] = self.crouch

			data['State']['depth'] = self.state['depth']
			data['State']['rgb'] = self.state['rgb']
			data['State']['mask'] = self.state['mask']
			data['State']['command'] = self.state['command']

			
			if world:
				data['State']["RightHandPose"]["RightHandWorldPos"] = self.state["RightHand"]["WorldLoc"]
				data['State']["LeftHandPose"]["LeftHandWorldPos"] = self.state["LeftHand"]["WorldLoc"]

				x,y,z = self.WorldToRel(self.state["RightHand"]["WorldLoc"]["X"], \
				self.state["RightHand"]["WorldLoc"]["Y"], self.state["RightHand"]["WorldLoc"]["Z"])
				self.state["RightHand"]["Loc"]["X"] = x
				self.state["RightHand"]["Loc"]["Y"] = y
				self.state["RightHand"]["Loc"]["Z"] = z

				x,y,z = self.WorldToRel(self.state["LeftHand"]["WorldLoc"]["X"], \
				self.state["LeftHand"]["WorldLoc"]["Y"], self.state["LeftHand"]["WorldLoc"]["Z"])
				self.state["LeftHand"]["Loc"]["X"] = x
				self.state["LeftHand"]["Loc"]["Y"] = y
				self.state["LeftHand"]["Loc"]["Z"] = z

			else:
				self.state["RightHand"]["WorldLoc"] = data['State']["RightHandPose"]["RightHandWorldPos"]
				self.state["LeftHand"]["WorldLoc"] = data['State']["LeftHandPose"]["LeftHandWorldPos"]

			# print self.state

		msg = dumps(data)
		msg = msg+"\n"
		self.client.send(msg.encode())

		# if reset, no need to receive feed back
		if not reset:
			data_recv = self.receive_msg()

			if data_recv and not data_recv['success']:
				self.state = copy.deepcopy(self.state_old)


			return data_recv

	def receive_msg(self):
		try:
			data_frame = {}

			object_head = self.server.getBuffer()
			assert(object_head == "Objects")
			object_data = loads(self.server.getBuffer())
			#print object_data
			
			if self.state['depth']:
				depth_head = self.server.getBuffer()
				assert(depth_head == "Depth")
				depth_data = np.load(BytesIO(self.server.getBuffer()))
				data_frame["depth"] = depth_data

			if self.state['rgb']:
				rgb_head = self.server.getBuffer()
				assert(rgb_head == "RGB")
				# try:
				rgb_data = np.load(BytesIO(self.server.getBuffer()))
				# rgb_data = Image.open(BytesIO(self.server.getBuffer()))
				# rgb_data = np.asarray(rgb_data)
				# except:
				# 	print("load error")
				# 	exit()
				# 	raise Exception("load error")
					
				data_frame["rgb"] = rgb_data
				# rgb_image = Image.fromarray(rgb_data)
				# rgb_image.show()
				# rgb_image.close()

			if self.state['mask']:
				mask_head = self.server.getBuffer()
				assert(mask_head == "object_mask")
				mask_data = np.load(BytesIO(self.server.getBuffer()))
				data_frame["object_mask"] = mask_data
		
			
			reward_head = self.server.getBuffer()
			assert(reward_head == "Reward")
			reward = float(self.server.getBuffer())

			done_head = self.server.getBuffer()
			assert(done_head == "Done")
			done = (self.server.getBuffer() == "1")

			success_head = self.server.getBuffer()
			assert(success_head == "Success")
			success = (self.server.getBuffer() == "1")

			if self.state['command']:
				command_head = self.server.getBuffer()
				assert(command_head == "Command")
				command_data = self.server.getBuffer()
				data_frame["command"] = command_data

				count_head = self.server.getBuffer()
				assert(count_head == "CommandCount")
				count_data = self.server.getBuffer()
				data_frame["count"] = count_data			


			data_frame["objects"] = object_data
			data_frame["frame"] = self.frame
			data_frame["reward"] = reward
			data_frame["done"] = done
			data_frame["success"] = success

			# for key in data_frame["objects"]:
			# 	print("objects: ", key)
			return data_frame

		except Exception as inst:
			print(type(inst))
			print(inst.args)
			return None

	def RelToWorld(self, x, y, z, theta):
		abs_length = np.sqrt(x**2+y**2)
		actor_theta = math.atan2(x, y)/math.pi*180
		actor_theta -= self.state["Actor"]["Rot"]["Yaw"]
		x_new = abs_length*math.sin(actor_theta/180*math.pi)+self.state["Actor"]["Loc"]["X"]
		y_new = abs_length*math.cos(actor_theta/180*math.pi)+self.state["Actor"]["Loc"]["Y"]
		z_new = z+self.state["Actor"]["Loc"]["Z"]

		theta_new = theta+self.state["Actor"]["Rot"]["Yaw"]
		if theta_new > 180:
			theta_new -= 360
		if theta_new < -180:
			theta_new += 360
		return x_new, y_new, z_new, theta_new

	def WorldToRel(self, x_new, y_new, z_new):
		z = z_new - self.state["Actor"]["Loc"]["Z"]
		x_temp = x_new - self.state["Actor"]["Loc"]["X"]
		y_temp = y_new - self.state["Actor"]["Loc"]["Y"]
		abs_length = (x_temp/(math.sin(math.atan2(x_temp, y_temp))+1e-5))**2
		alpha = (math.atan2(x_temp, y_temp)*180/math.pi+self.state["Actor"]["Rot"]["Yaw"])
		rate = math.tan(alpha/180*math.pi)
		x = np.sqrt(abs_length*rate**2/(rate**2+1))
		y = x/rate

		return x, y, z

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

	def Crouch(self):
		if self.crouch == False:
			self.crouch = True
			self.Move("LeftHand", 2, scale=-8)
			self.Move("LeftHand", 1, scale=-2)
			self.Rotate("LeftHand", 1, scale=1.0)
			self.state["LeftHand"]["NeutralLoc"]["Z"] -= 80
			self.state["LeftHand"]["NeutralLoc"]["Y"] -= 20
			self.state["LeftHand"]["NeutralRot"]["Yaw"] += 90

			#self.step("RightHandMoveDown", scale = 8.5)
			self.Move("RightHand", 2, scale=-8.5)
			self.state["RightHand"]["NeutralLoc"]["Z"] -= 85

			# self.state["Head"]["Rot"]["Roll"] = -self.state["Head"]["Rot"]["Pitch"]
			# self.state["Head"]["Rot"]["Pitch"] = 0

		data_frame = None
		i = 0
		while data_frame == None and i < self.retry_num:
			data_frame = self.send_tf(world=False)
			i += 1

	
		if data_frame == None:
			print("Connection break. Manual shutdown.") 

		self.frame += 1
		self.data = data_frame

		return data_frame

	def Standup(self):
		if self.crouch == True:
			self.crouch = False
			self.Move("LeftHand", 2, scale=8)
			self.Move("LeftHand", 1, scale=2)
			self.Rotate("LeftHand", 1, scale=-1.0)
			self.state["LeftHand"]["NeutralLoc"]["Z"] += 80
			self.state["LeftHand"]["NeutralLoc"]["Y"] += 20
			self.state["LeftHand"]["NeutralRot"]["Yaw"] -= 90

			self.Move("RightHand", 2, scale=8.5)
			self.state["RightHand"]["NeutralLoc"]["Z"] += 85

			# self.state["Head"]["Rot"]["Pitch"] = -self.state["Head"]["Rot"]["Roll"]
			# self.state["Head"]["Rot"]["Roll"] = 0

		data_frame = None
		i = 0
		while data_frame == None and i < self.retry_num:
			data_frame = self.send_tf(world=False)
			i += 1

		if data_frame == None:
			print("Connection break. Manual shutdown.") 

		self.frame += 1
		self.data = data_frame

		return data_frame

	def NoOp(self):
		data_frame = None
		i = 0
		while data_frame == None and i < self.retry_num:
			data_frame = self.send_tf(world=False)
			i += 1

		if data_frame == None:
			print("Connection break. Manual shutdown.") 

		self.frame += 1
		self.data = data_frame

		return data_frame

	def Walk(self, loc_final, speed=5, anim_speed=40):
		loc_now = self.state["Actor"]["Loc"]
		dif = np.array([loc_final[key] - loc_now[key] for key in loc_now])
		dif_len = np.linalg.norm(dif)

		times = np.ceil(dif_len/speed)
		move_unit = dif/times

		for i in np.arange(times):
			cnt = 0
			for key in loc_now:
				loc_now[key] += move_unit[cnt]
				cnt += 1
			
			# print loc_now

			data_frame = None
			i = 0
			while data_frame == None and i < self.retry_num:
				data_frame = self.send_tf(anim_speed=anim_speed)
				i += 1

			if data_frame == None:
				print("Connection break. Manual shutdown.") 

			self.frame += 1
			self.data = data_frame

		return data_frame

	def GrabObject(self, entity, actor_name, comp_name):
		self.state[entity]["ActorName"] = actor_name
		self.state[entity]["CompName"] = comp_name
		self.state[entity]["Grab"] = True
		a = self.send_tf(world=True)
		# set grab to false to avoid checking too many times
		self.state[entity]["Grab"] = False
		return a

	def ReleaseObject(self, entity):
		self.state[entity]["Release"] = True
		a = self.send_tf(world=True)
		self.state[entity]["Release"] = False
		self.state[entity]["ActorName"] = ""
		self.state[entity]["CompName"] = ""
		return a

	def MoveToObject(self, entity, actor_name, comp_name, \
		speed=SPEED, closest=False):
		ActorLoc = copy.deepcopy(self.state["Actor"]["Loc"])

		if self.crouch:
			ActorLoc["Z"] += 50
		else:
			ActorLoc["Z"] += 150

		start_time = time.time()
		if closest:
			min_dist = 1e8
			min_loc = None

			for actor in self.data["objects"]:
				if actor_name not in self.data["objects"][actor]["ActorType"]:
					continue
				for comp in self.data["objects"][actor]:
					if comp == "ActorType":
						continue
					if comp_name != None and comp != comp_name:
						continue
					CompLoc = self.data["objects"][actor][comp]["Loc"]
					dist = 0
					for key in CompLoc:
						dist += (CompLoc[key]-ActorLoc[key])**2
					if dist < min_dist:
						min_dist = dist
						min_loc = CompLoc

			loc_final = min_loc

		else:
			loc_final = self.data["objects"][actor_name][comp_name]["Loc"]

		# print "take time to find closest", time.time()-start_time
		
		return self.MoveToWorld(entity, loc_final, speed)

	def MoveAndGrabObject(self, entity, actor_name, comp_name, \
		speed=SPEED, closest=False):
		ActorLoc = copy.deepcopy(self.state["Actor"]["Loc"])
		if self.crouch:
			ActorLoc["Z"] += 50
		else:
			ActorLoc["Z"] += 150

		if closest:
			min_dist = 1e8
			min_loc = None
			min_actor = None
			min_comp = None

			for actor in self.data["objects"]:
				if actor_name not in self.data["objects"][actor]["ActorType"]:
					continue
				for comp in self.data["objects"][actor]:
					if comp == "ActorType":
						continue

					if comp_name != None and comp != comp_name:
						continue
					CompLoc = self.data["objects"][actor][comp]["Loc"]
					dist = 0
					for key in CompLoc:
						dist += (CompLoc[key]-ActorLoc[key])**2
					if dist < min_dist:
						min_dist = dist
						min_loc = CompLoc
						min_actor = actor
						min_comp = comp

			loc_final = min_loc

		else:
			loc_final = self.data["objects"][actor_name][comp_name]["Loc"]
		
		self.MoveToWorld(entity, loc_final, speed)
		self.GrabObject(entity, min_actor, "ContainerMesh")
		return self.MoveToNeutral(entity, speed)

	def MoveToNeutral(self, entity, speed=SPEED):
		loc = self.state[entity]["NeutralLoc"]
		x_new, y_new, z_new, _ = self.RelToWorld(loc["X"], loc["Y"], loc["Z"], 0)
		loc_final = {"X":x_new, "Y":y_new, "Z":z_new}
		return self.MoveToWorld(entity, loc_final, speed)		

	def MoveContactToWorld(self, entity, loc_final, speed=SPEED):
		eps = 1e-3
		
		loc_now =  self.state[entity]["WorldLoc"]
		dif = np.array([loc_final[key] - loc_now[key] for key in loc_now])
		dif_len = np.linalg.norm(dif)

		times = np.ceil(dif_len/speed+eps)
		move_unit = dif/times

		for i in np.arange(times):
			cnt = 0
			for key in loc_now:
				loc_now[key] += move_unit[cnt]
				cnt += 1
			# print loc_now

			data_frame = None
			i = 0
			while data_frame == None and i < self.retry_num:
				data_frame = self.send_tf(world=True)
				i += 1

			if data_frame == None:
				print("Connection break. Manual shutdown.") 

			self.frame += 1
			self.data = data_frame

		return data_frame

	def MoveContactToObject(self, entity, contact_name, \
			actor_name, comp_name, speed=SPEED):
		grab_actor_name = self.state[entity]["ActorName"]
		temp = [
			self.data["objects"][actor_name][comp_name]["Loc"][key] - \
			self.data['objects'][grab_actor_name][contact_name]["Loc"][key] + \
			self.state[entity]["WorldLoc"][key] 
			for key in self.data["objects"][actor_name][comp_name]["Loc"]
		]
		loc_final = {"Y": temp[0], "X":temp[1], "Z":temp[2]}
		# print self.data["objects"][actor_name][comp_name]["Loc"]
		# print self.data['objects'][grab_actor_name][contact_name]["Loc"]
		# print self.state[entity]["WorldLoc"] 
		# print loc_final
		return self.MoveContactToWorld(entity, loc_final, speed)

	def MoveToWorld(self, entity, loc_final, speed=SPEED):
		eps = 1e-3
		loc_now = self.state[entity]["WorldLoc"]
		if loc_final == None:
			loc_final = loc_now
		dif = np.array([loc_final[key] - loc_now[key] for key in loc_now])
		dif_len = np.linalg.norm(dif)

		times = np.ceil(dif_len/speed+eps)
		move_unit = dif/times

		for i in np.arange(times):
			cnt = 0
			for key in loc_now:
				loc_now[key] += move_unit[cnt]
				cnt += 1
			
			# print loc_now

			data_frame = None
			i = 0
			while data_frame == None and i < self.retry_num:
				data_frame = self.send_tf(world=True)
				i += 1

			if data_frame == None:
				print("Connection break. Manual shutdown.") 

			self.frame += 1
			self.data = data_frame

		return data_frame

	def CutObject(self, entity, actor_name, speed=5):
		self.MoveToObject(entity,"Knife","StaticMeshComponent0")
		self.GrabObject(entity,"Knife","StaticMeshComponent0")
		self.MoveToNeutral(entity)
		self.step(entity+"TwistLeft", scale=5)
		self.MoveContactToObject(entity,"CutPoint", actor_name, "Fruit")
		self.MoveToNeutral(entity)
		self.MoveContactToObject(entity,"CutPoint", actor_name, "Fruit")
		self.MoveToNeutral(entity)
		self.MoveContactToObject(entity,"CutPoint", actor_name, "Fruit")
		self.MoveToNeutral(entity)
		self.MoveContactToObject(entity,"CutPoint", actor_name, "ProceduralMeshComponent_0")
		self.MoveToNeutral(entity)
		self.MoveContactToObject(entity,"CutPoint", actor_name, "ProceduralMeshComponent_0")
		self.MoveToNeutral(entity)
		self.MoveContactToObject(entity,"CutPoint", actor_name, "ProceduralMeshComponent_0")
		res = self.MoveToNeutral(entity)
		return res

	def OpenBottle(self, entity1, entity2, opener_name, bottle_name, speed=5):
		self.MoveToObject(entity1, opener_name, "StaticMeshComponent0", speed)
		self.GrabObject(entity1, opener_name, "StaticMeshComponent0")
		self.MoveToNeutral(entity1, speed)
		self.MoveToObject(entity2, bottle_name, "grabpoint", speed)
		self.GrabObject(entity2, bottle_name, "ContainerMesh")
		self.MoveToNeutral(entity2, speed)
		self.MoveContactToObject(entity1, "Box", bottle_name, "Box", speed)
		res = self.MoveToNeutral(entity1, speed)
		return res

	def ToastBread(self, entity, bread_name, toaster_name, speed=5):
		self.MoveToObject(entity, bread_name, "grabpoint", speed)
		self.GrabObject(entity, bread_name, "StaticMeshComponent0")
		self.MoveToNeutral(entity, speed)
		self.step(entity+"RotateUp", scale=5)
		self.MoveContactToObject(entity, "StaticMeshComponent0", toaster_name, "PutBox", speed)
		self.ReleaseObject(entity)
		res = self.MoveToNeutral(entity, speed)
		return res

	def GoToPos(self, PosName):
		self.state["Actor"] = tool_pos[self.state["scene"]][PosName]["Actor"]
		data_frame = None
		i = 0
		while data_frame == None and i < self.retry_num:
			data_frame = self.send_tf()
			i += 1

		if data_frame == None:
			print("Go to Pos. Connection break. Manual shutdown.") 

		self.frame += 1
		self.data = data_frame

		return data_frame


	def step(self, action, world=False, scale=1.0):
		theta = self.state["Actor"]["Rot"]["Yaw"]/180*math.pi
		x = math.cos(theta)*scale
		y = math.sin(theta)*scale
		self.state_old = copy.deepcopy(self.state)
		# print x
		# print y
		st = time.time()
		if action == "ActorMoveForward":
			self.Move("Actor", 0, scale=x)
			self.Move("Actor", 1, scale=y)
		elif action == "MoveToDoor":
			self.MoveToObject("RightHand", "door", None, speed=10000, closest=True)
			self.MoveToNeutral("RightHand", speed=10000)
			print("MoveToDoor", time.time()-st)
		elif action == "GrabCup":
			self.MoveAndGrabObject("RightHand", "Pour", "grabpoint", speed=10000, \
				closest=True)
			print("GrabCup", time.time()-st)
		elif action == "MoveToCoffeMaker":
			self.MoveToObject("RightHand", "Coffe", "PlaceForCup", speed=10000, closest=True)
			self.ReleaseObject("RightHand")
			self.MoveToNeutral("RightHand", speed=10000)
			print("MoveToCoffeMaker", time.time()-st)
		elif action == "OpenCoffeMaker":
			self.MoveToObject("RightHand", "Coffe", "PowerButton", speed=10000, closest=True)
			self.MoveToObject("RightHand", "Coffe", "PourCoffeeButton", speed=10000, closest=True)
			self.MoveToNeutral("RightHand", speed=10000)
			print("OpenCoffeMaker", time.time()-st)
		elif action == "MoveToNeutral":
			self.MoveToNeutral("RightHand", speed=10000)
		elif action == "Crouch":
			self.Crouch()
			print("crouch", time.time()-st)
		elif action == "Standup":
			self.Standup()
			print("Standup", time.time()-st)
		elif action == "ActorMoveBackward":
			self.Move("Actor", 0, scale=-x)
			self.Move("Actor", 1, scale=-y)
		elif action == "ActorMoveRight":
			self.Move("Actor", 0, scale=-y)
			self.Move("Actor", 1, scale=x)
		elif action == "ActorMoveLeft":
			self.Move("Actor", 0, scale=y)
			self.Move("Actor", 1, scale=-x)
		elif action == "ActorRotateRight":
			self.Rotate("Actor", 1, scale=1)
		elif action == "ActorRotateLeft":
			self.Rotate("Actor", 1, scale=-1)
		elif action == "LeftHandMoveForward":
			self.Move("LeftHand", 0, scale=scale)
		elif action == "LeftHandMoveBackward":
			self.Move("LeftHand", 0, scale=-scale)
		elif action == "LeftHandMoveRight":
			self.Move("LeftHand", 1, scale=scale)
		elif action == "LeftHandMoveLeft":
			self.Move("LeftHand", 1, scale=-scale)
		elif action == "LeftHandMoveUp":
			self.Move("LeftHand", 2, scale=scale)
		elif action == "LeftHandMoveDown":
			self.Move("LeftHand", 2, scale=-scale)
		elif action == "LeftHandRotateRight":
			self.Rotate("LeftHand", 1, scale=1.0/5*scale)
		elif action == "LeftHandRotateLeft":
			self.Rotate("LeftHand", 1, scale=-1.0/5*scale)
		elif action == "LeftHandRotateUp":
			self.Rotate("LeftHand", 0, scale=1.0/5*scale)
		elif action == "LeftHandRotateDown":
			self.Rotate("LeftHand", 0, scale=-1.0/5*scale)
		elif action == "LeftHandTwistRight":
			self.Rotate("LeftHand", 2, scale=1.0/5*scale)
		elif action == "LeftHandTwistLeft":
			self.Rotate("LeftHand", 2, scale=-1.0/5*scale)
		elif action == "RightHandMoveForward":
			self.Move("RightHand", 0, scale=scale)
		elif action == "RightHandMoveBackward":
			self.Move("RightHand", 0, scale=-scale)
		elif action == "RightHandMoveRight":
			self.Move("RightHand", 1, scale=scale)
		elif action == "RightHandMoveLeft":
			self.Move("RightHand", 1, scale=-scale)
		elif action == "RightHandRotateRight":
			self.Rotate("RightHand", 1, scale=1.0/5*scale)
		elif action == "RightHandRotateLeft":
			self.Rotate("RightHand", 1, scale=-1.0/5*scale)
		elif action == "RightHandRotateUp":
			self.Rotate("RightHand", 0, scale=1.0/5*scale)
		elif action == "RightHandRotateDown":
			self.Rotate("RightHand", 0, scale=-1.0/5*scale)
		elif action == "RightHandTwistRight":
			self.Rotate("RightHand", 2, scale=1.0/5*scale)
		elif action == "RightHandTwistLeft":
			self.Rotate("RightHand", 2, scale=-1.0/5*scale)
		elif action == "RightHandMoveUp":
			self.Move("RightHand", 2, scale=scale)
		elif action == "RightHandMoveDown":
			self.Move("RightHand", 2, scale=-scale)
		elif action == "LookUp":
			self.Rotate("Head", 0, scale=1.0/3*scale)
		elif action == "LookDown":
			self.Rotate("Head", 0, scale=-1.0/3*scale)
		elif action == "LeftHandGrab":
			self.state["LeftHand"]["Grab"] = True
		elif action == "LeftHandRelease":
			self.state["LeftHand"]["Grab"] = False
		elif action == "RightHandGrab":
			self.state["RightHand"]["Grab"] = True
		elif action == "RightHandRelease":
			self.state["RightHand"]["Grab"] = False
		else:
			print("Not a valid command")

		data_frame = None
		i = 0
		while data_frame == None and i < self.retry_num:
			data_frame = self.send_tf(world=world)
			i += 1

		if data_frame == None:
			print("Connection break. Manual shutdown.") 

		self.frame += 1
		self.data = data_frame

		return data_frame

	





 


