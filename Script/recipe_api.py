import math
#import Image
from PIL import Image
import numpy as np
from DiscreteAgent import DiscreteAgent
import time
import os


class recipe(object): 
	def __init__(self):
		self.PosList = ["Orig", "Grater", "Sauce", "Knife", "Peeler", "Juicer",
					"Oven", "Stove", "Fridge"]

		self.LeftHandList = ["Lettuce", "Tomato", "Cucumber", "Eggplant", "cheese", "Cheese_sliced", "Dough", "Onion", "Potato"]
		self.StaticMeshList = ["BreadBP", "Dough", "Cheese_sliced"]
		self.CenterList = ["BreadBP", "Dough"]
		self.BoxTwoList = []
		self.BoxList = ["Board"]
		self.FoodPointList = ["Stove"]
		self.ContainerMesh = ["Pot"]

		self.UseList = ["Peeler", "Knife", "Juicer", "Cup", "Oven", "SauceBottle",
		"grater"]
		self.OpenList = ["FridgeDoorDown", "Stove"]

		self.IngredList = ["Lettuce", "Tomato", "Cucumber", "Eggplant", "Onion", "cheese", "Cheese_sliced", "Dough", \
			"Beef", "Chicken", "BreadBP", "Lemon", "Mango", "Kiwi", "Peach", "Apple", "Orange", "Ham", "Turkey", \
			"Salami", "Potato"]
		self.ContList = ["Fridge", "Plate", "Plate2", "Hand", "BreadBP", "Dough", "Pot", "Board", "Cup","SauceBottle"]
		self.CutableList = ["Lettuce", "Tomato", "Cucumber", "Eggplant", "Onion", "Potato", "Lemon", "Mango", "Kiwi", \
			"Peach", "Apple", "Orange"]
		self.PeelableList = ["Apple", "Cucumber", "Kiwi", "Mango", "Orange", "Peach", "Potato"]
		self.JuiceList = ["Cucumber", "Tomato", "Apple", "Kiwi", "Mango", "Orange", "Peach", "Lemon"]
		self.GratableList = ["cheese"]

		self.env = DiscreteAgent({"Name":"Agent1", \
					"Actor":{"Loc":{"X":0,"Y":0,"Z":0},\
						"Rot":{"Pitch":0.0,"Yaw":0.0,"Roll":0.0}}, \

					"LeftHand":{"Grab": False, "Release": False, "ActorName":"", "CompName":"",
						"NeutralLoc":{"X":40,"Y":-10,"Z":120},"NeutralRot":{"Pitch":0.0,"Yaw":0.0,"Roll":0.0},
						"Loc":{"X":40,"Y":-10,"Z":120},"Rot":{"Pitch":0.0,"Yaw":0.0,"Roll":0.0},
						"WorldLoc":{"X":0,"Y":0,"Z":0}},\

					"RightHand":{"Grab": False, "Release": False, "ActorName":"", "CompName":"",\
						"NeutralLoc":{"X":40,"Y":10,"Z":120}, "NeutralRot":{"Pitch":0.0,"Yaw":0.0,"Roll":0.0}, \
						"Loc":{"X":40,"Y":10,"Z":120}, "Rot":{"Pitch":0.0,"Yaw":0.0,"Roll":0.0}, \
						"WorldLoc":{"X":0,"Y":0,"Z":0}},

					"Head": {"Rot":{"Pitch":-10,"Yaw":0.0,"Roll":0}},
					"rgb": False, "depth": False, "mask": False, "command":True,
					"scene": "2"
					}
		)
		self.ObjDict = {}
		self.InitDict()
		self.a = self.env.start()

	def InitDict(self):
		for obj in self.IngredList:
			self.ObjDict[obj] = {"Pos": "Fridge", "Cut":False, "Peel":False, "Juice":False, "Cook":False}
		self.ObjDict['Sauce'] = {"Pos": "SauceBottle", "Cut":False, "Peel":False, "Juice":False, "Cook":False}


	def GoTo(self,loc, folder_name="", count=[0], f_label="", f_fluent=""):
		if loc not in self.PosList:
			return "Location not in list"
			 

		self.a = self.env.GoToPos(loc)
		# if folder_name:
		# 	self.env.state['rgb'] = True
		# 	self.a = self.env.send_tf()
			
		# 	self.env.state['rgb'] = False
		# 	count[0] += 1
		# 	flag = True
		# 	while flag:
		# 		try:
		# 			rgb_image = Image.fromarray(self.a['rgb'])
		# 			rgb_image.save(folder_name+"/"+str(count[0])+".jpg")
		# 			if os.path.exists("static/img/temp.jpg"):
  # 						os.remove("static/img/temp.jpg")
		# 			rgb_image.save("static/img/temp.jpg")
		# 			flag = False
		# 		except:
		# 			pass

		if f_label:
			f_label.write("Goto ")
			f_label.write(loc)
			f_label.write("\n")

		if f_fluent:
			f_fluent.write(str(self.ObjDict))
			f_fluent.write("\n")
		return "Success"
		
	def RecordSuccess(self,f_label=""):
		if f_label:
			f_label.write("finshed the task")
			


	def Take(self,obj, folder_name="", count=[0], f_label="", f_fluent=""):
		# TODO: add constraint that one hand can take things at a time
		if obj not in self.env.data['objects']:
			return "not near the object"+obj+", fail to take"
			

		if obj in self.LeftHandList:
			hand = "LeftHand"
		else:
			hand = "RightHand"

		if obj in self.CenterList:
			# if in center list, move to center first
			comp = "Center"
		elif obj in self.StaticMeshList:
			comp = "StaticMeshComponent0"
		else:
			comp = "ProcMesh"

		self.a = self.env.MoveToObject(hand, obj, comp)

		if obj in self.StaticMeshList:
			comp = "StaticMeshComponent0"
		else:
			comp = "ProcMesh"

		self.a = self.env.GrabObject(hand, obj, comp)
		self.a = self.env.MoveToNeutral(hand)

		self.ObjDict[obj]['Pos'] = "Hand"

		# if folder_name:
		# 	self.env.state['rgb'] = True
		# 	self.a = self.env.send_tf()
			
		# 	self.env.state['rgb'] = False
		# 	count[0] += 1
		# 	flag = True
		# 	while flag:
		# 		try:
		# 			rgb_image = Image.fromarray(self.a['rgb'])
		# 			rgb_image.save(folder_name+"/"+str(count[0])+".jpg")
		# 			if os.path.exists("static/img/temp.jpg"):
  # 						os.remove("static/img/temp.jpg")
		# 			rgb_image.save("static/img/temp.jpg")
		# 			flag = False
		# 		except:
		# 			pass

		if f_label:
			f_label.write("Take ")
			f_label.write(obj)
			f_label.write("\n")

		if f_fluent:
			f_fluent.write(str(self.ObjDict))
			f_fluent.write("\n")

		return "Success"

	def PlaceTo(self,obj, folder_name="", count=[0], f_label="", f_fluent=""):
		if self.env.state["LeftHand"]["ActorName"]:
			hand = "LeftHand"
			actor_in_hand = self.env.state["LeftHand"]["ActorName"]
			comp_in_hand = self.env.state["LeftHand"]["CompName"]
		elif self.env.state["RightHand"]["ActorName"]:
			hand = "RightHand"
			actor_in_hand = self.env.state["RightHand"]["ActorName"]
			comp_in_hand = self.env.state["RightHand"]["CompName"]
		else:
			hand = ""
			print "nothing to place"
			return 

		if actor_in_hand in self.CenterList:
			# if in center list, move to center first
			comp = "Center"
		elif actor_in_hand in self.StaticMeshList:
			comp = "StaticMeshComponent0"
		else:
			comp = "ProcMesh"

		if obj in self.BoxTwoList:
			comp_place = "Box2"
		elif obj in self.BoxList:
			comp_place = "Box"
		elif obj in self.FoodPointList:
			comp_place = "FoodPoint"
		elif obj in self.CenterList:
			comp_place = "Center"
		elif obj in self.ContainerMesh:
			comp_place = "ContainerMesh"
		else:
			comp_place = "StaticMeshComponent0"

		self.a = self.env.MoveContactToObject(hand, comp, obj, comp_place)
		self.a = self.env.ReleaseObject(hand)
		self.a = self.env.MoveToNeutral(hand)

		if obj in self.ContList:
			self.ObjDict[actor_in_hand]["Pos"] = obj


		# if folder_name:
		# 	self.env.state['rgb'] = True
		# 	self.a = self.env.send_tf()
			
		# 	self.env.state['rgb'] = False
		# 	count[0] += 1
		# 	flag = True
		# 	while flag:
		# 		try:
		# 			rgb_image = Image.fromarray(self.a['rgb'])
		# 			rgb_image.save(folder_name+"/"+str(count[0])+".jpg")
		# 			if os.path.exists("static/img/temp.jpg"):
  # 						os.remove("static/img/temp.jpg")
		# 			rgb_image.save("static/img/temp.jpg")
		# 			flag = False
		# 		except:
		# 			pass

		if f_label:
			f_label.write("Placeto ")
			f_label.write(obj)
			f_label.write("\n")

		if f_fluent:
			f_fluent.write(str(self.ObjDict))
			f_fluent.write("\n")
		return "Success"

	def Use(self,tool, folder_name="", count=[0], f_label="", f_fluent=""):
		if self.env.state["LeftHand"]["ActorName"]:
			hand = "LeftHand"
			actor_in_hand = self.env.state["LeftHand"]["ActorName"]
			comp_in_hand = self.env.state["LeftHand"]["CompName"]
		elif self.env.state["RightHand"]["ActorName"]:
			hand = "RightHand"
			actor_in_hand = self.env.state["RightHand"]["ActorName"]
			comp_in_hand = self.env.state["RightHand"]["CompName"]
		else:
			hand = ""
			actor_in_hand = ""
			comp_in_hand = ""

		if not tool == "Oven" and tool not in self.env.data["objects"]:
			print(tool)
			return "not near the tool"+ tool
			 

		if tool == "Knife":
			if not hand:
				return "nothing to cut"
				 
			self.a = self.env.MoveContactToObject(hand, "ProcMesh", "Board", "Box")
			self.a = self.env.ReleaseObject(hand)
			self.a = self.env.MoveToNeutral(hand) 
			self.a = self.env.MoveToObject("RightHand", "Knife", "grabpoint")
			self.a = self.env.GrabObject("RightHand", "Knife", "StaticMeshComponent0")
			self.a = self.env.MoveToNeutral("RightHand")
			self.a = self.env.step("RightHandTwistLeft", scale=5)
			self.a = self.env.step("RightHandMoveLeft", scale=2.7)
			self.a = self.env.MoveContactToObject("RightHand", "CutPoint", actor_in_hand, "ProcMesh")
			self.a = self.env.step("RightHandMoveUp", scale=2)
			self.a = self.env.MoveToNeutral("RightHand")
			self.a = self.env.step("RightHandTwistRight", scale=5)
			self.a = self.env.MoveContactToObject("RightHand", "grabpoint", "282Main_18", "Box")
			self.a = self.env.ReleaseObject("RightHand")
			self.a = self.env.MoveToNeutral("RightHand")
			self.ObjDict[actor_in_hand]["Pos"] = "Board"
			if actor_in_hand in self.CutableList:
				self.ObjDict[actor_in_hand]["Cut"] = True

		elif tool == "Peeler":
			if hand == "LeftHand":
				return "LeftHand not empty"
				
			if not hand:
				return "nothing to peel"
				
			self.a = self.env.MoveToObject("LeftHand", "Peeler", "grabpoint")
			self.a = self.env.GrabObject("LeftHand", "Peeler", "StaticMeshComponent0")
			self.a = self.env.MoveToNeutral("LeftHand")
			self.a = self.env.MoveContactToObject("LeftHand","Blade", actor_in_hand, "Skin")
			self.a = self.env.MoveToNeutral("LeftHand")
			self.a = self.env.MoveContactToObject("LeftHand", "grabpoint", "282Main3", "Box")
			self.a = self.env.ReleaseObject("LeftHand")
			self.a = self.env.MoveToNeutral("LeftHand")
			if actor_in_hand in self.PeelableList:
				self.ObjDict[actor_in_hand]["Peel"] = True

		elif tool == "grater":
			if hand == "RightHand":
				return "RightHand not empty"
				 
			if not hand:
				return "nothing to grate"
				 
			self.a = self.env.MoveToObject("RightHand", "grater", "grabpoint")
			self.a = self.env.GrabObject("RightHand", "grater", "StaticMeshComponent0")
			self.a = self.env.MoveToNeutral("RightHand")
			self.a = self.env.MoveContactToObject("RightHand", "StaticMeshComponent0", actor_in_hand, "ProcMesh")
			self.a = self.env.MoveToNeutral("RightHand")
			self.a = self.env.MoveContactToObject("RightHand", "StaticMeshComponent0", "284Main_2", "Box")
			self.a = self.env.ReleaseObject("RightHand")
			self.a = self.env.MoveToNeutral("RightHand")
			self.a = self.env.MoveContactToObject("LeftHand", "ProcMesh", "284Main_2", "Box1")
			self.a = self.env.ReleaseObject("LeftHand")
			self.a = self.env.MoveToNeutral("LeftHand")
			if actor_in_hand in self.GratableList:
				for thing in self.ObjDict:
					if thing in self.ContList and self.ObjDict[thing]['Pos'] == "Plate":
						self.ObjDict[actor_in_hand]["Pos"] = thing
						break

		elif tool == "SauceBottle":
			if hand == "RightHand":
				return "RightHand not empty"
				 
			self.a = self.env.MoveToObject("RightHand", "SauceBottle", "StaticMeshComponent0")
			self.a = self.env.GrabObject("RightHand","SauceBottle","StaticMeshComponent0")
			self.a = self.env.MoveToNeutral("RightHand")
			self.a = self.env.MoveContactToObject("RightHand","StaticMeshComponent0", "Plate2","Box2")
			self.a = self.env.step("RightHandTwistLeft", scale=6)
			self.a = self.env.step("RightHandTwistRight", scale=6)
			self.a = self.env.MoveToNeutral("RightHand")
			self.a = self.env.MoveContactToObject("RightHand","StaticMeshComponent0", "281Main_5","Box")
			self.a = self.env.ReleaseObject("RightHand")
			self.a = self.env.MoveToNeutral("RightHand")
			for thing in self.ObjDict:
				if thing in self.ContList and self.ObjDict[thing]['Pos'] == "Plate2":
					self.ObjDict["Sauce"]["Pos"] = thing
					break

		elif tool == "Juicer":
			if not hand:
				return "nothing to juice"
				
			self.a = self.env.GoToPos("Juicer")
			self.a = self.env.MoveContactToObject(hand,"ProcMesh", "Juicer", "SqueezeBox")
			self.a = self.env.ReleaseObject(hand)
			self.a = self.env.MoveToNeutral(hand)
			if actor_in_hand in self.JuiceList:
				self.ObjDict[actor_in_hand]["Pos"] = "Cup"
				self.ObjDict[actor_in_hand]["Juice"] = True

		elif tool == "Cup":
			if hand == "LeftHand":
				return "LeftHand not empty"
				
			self.a = self.env.MoveToObject("LeftHand", "Cup", "grabpoint")
			self.a = self.env.GrabObject("LeftHand", "Cup", "ContainerMesh")
			self.a = self.env.MoveToNeutral("LeftHand")
			self.a = self.env.GoToPos("Stove")
			self.a = self.env.MoveContactToObject("LeftHand","ContainerMesh", "Pot","PlaceForBlender")
			self.a = self.env.step("LeftHandTwistRight", scale=5)
			self.a = self.env.step("LeftHandTwistLeft", scale=5)
			self.a = self.env.MoveToNeutral("LeftHand")
			self.a = self.env.GoToPos("Juicer")
			self.a = self.env.MoveContactToObject("LeftHand","ContainerMesh", "Juicer","PlaceForCup")
			self.a = self.env.ReleaseObject("LeftHand")
			self.a = self.env.MoveToNeutral("LeftHand")
			for thing in self.ObjDict:
				if self.ObjDict[thing]['Pos'] == "Cup":
					self.ObjDict[thing]['Pos'] = "Pot"

		elif tool == "Oven":
			if not hand:
				return "nothing to place into oven"
				
			if hand == "LeftHand":
				open_stove_hand = "RightHand"
			else:
				open_stove_hand = "LeftHand"
			self.a = self.env.Crouch()
			self.a = self.env.MoveToObject(open_stove_hand, "StoveDoor", "Box")
			self.a = self.env.MoveToNeutral(open_stove_hand)
			self.a = self.PlaceTo("Stove")
			self.a = self.Take(actor_in_hand)
			self.a = self.env.Standup()
			self.a = self.env.Standup()
			for thing in self.ObjDict:
				if self.ObjDict[thing]['Pos'] == "Hand" or self.ObjDict[thing]['Pos'] == actor_in_hand:
					self.ObjDict[thing]['Cook'] = True

		# if folder_name:
		# 	self.env.state['rgb'] = True
		# 	self.a = self.env.send_tf()
			
		# 	self.env.state['rgb'] = False
		# 	count[0] += 1
		# 	flag = True
		# 	while flag:
		# 		try:
		# 			rgb_image = Image.fromarray(self.a['rgb'])
		# 			rgb_image.save(folder_name+"/"+str(count[0])+".jpg")
		# 			if os.path.exists("static/img/temp.jpg"):
  # 						os.remove("static/img/temp.jpg")
		# 			rgb_image.save("static/img/temp.jpg")
		# 			flag = False
		# 		except:
		# 			pass

		if f_label:
			f_label.write("Use ")
			f_label.write(tool)
			f_label.write("\n")

		if f_fluent:
			f_fluent.write(str(self.ObjDict))
			f_fluent.write("\n")

		return "Success"

	def Open(self,tool, folder_name="", count=[0], f_label="", f_fluent=""):
		if self.env.state["LeftHand"]["ActorName"]:
			hand = "LeftHand"
			actor_in_hand = self.env.state["LeftHand"]["ActorName"]
			comp_in_hand = self.env.state["LeftHand"]["CompName"]
		elif self.env.state["RightHand"]["ActorName"]:
			hand = "RightHand"
			actor_in_hand = self.env.state["RightHand"]["ActorName"]
			comp_in_hand = self.env.state["RightHand"]["CompName"]
		else:
			hand = ""

		if tool == "Fridge":
			self.a = self.env.MoveToObject("RightHand", "FridgeDoorDown", "Box")
			self.a = self.env.MoveToNeutral("RightHand")
		elif tool == "Stove":
			self.a = self.env.MoveToObject("LeftHand","Stove","switch")
			self.a = self.env.MoveToObject("LeftHand","Stove","switch")
			self.a = self.env.MoveToNeutral("LeftHand")
			for thing in self.ObjDict:
				if self.ObjDict[thing]['Pos'] == "Pot":
					self.ObjDict[thing]['Cook'] = True

		# if folder_name:
		# 	self.env.state['rgb'] = True
		# 	self.a = self.env.send_tf()
			
		# 	self.env.state['rgb'] = False
		# 	count[0] += 1
		# 	flag = True
		# 	while flag:
		# 		try:
		# 			rgb_image = Image.fromarray(self.a['rgb'])
		# 			rgb_image.save(folder_name+"/"+str(count[0])+".jpg")
		# 			if os.path.exists("static/img/temp.jpg"):
  # 						os.remove("static/img/temp.jpg")
		# 			rgb_image.save("static/img/temp.jpg")
		# 			flag = False
		# 		except:
		# 			pass


		if f_label:
			f_label.write("Open ")
			f_label.write(tool)
			f_label.write("\n")

		if f_fluent:
			f_fluent.write(str(self.ObjDict))
			f_fluent.write("\n")

		return "Success"
	def LookUp(self, f_label=""):
		self.a = self.env.step("LookUp")
		if f_label:
			f_label.write("look up")
			f_label.write("\n")
		return "Success"
	def LookDown(self, f_label=""):
		self.a = self.env.step("LookDown")
		if f_label:
			f_label.write("look down")
			f_label.write("\n")
		return "Success"














