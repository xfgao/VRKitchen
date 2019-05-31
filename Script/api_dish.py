import math
#import Image
from PIL import Image
import numpy as np
from DiscreteAgent import DiscreteAgent

PosList = ["Orig", "Grater", "SauceBottle", "Knife", "Peeler", "Juicer",
			"Oven", "Stove", "Fridge"]

LeftHandList = ["Lettuce", "Tomato", "Cucumber", "Eggplant", "Cheese", "Cheese_sliced", "Dough", "Onion", "Potato"]
StaticMeshList = ["BreadBP", "Dough", "Cheese_sliced"]
CenterList = ["BreadBP", "Dough"]
BoxTwoList = []
BoxList = ["CutBoard"]
FoodPointList = ["Stove"]
ContainerMesh = ["Pot"]

IngredList = ["Lettuce", "Tomato", "Cucumber", "Eggplant", "Onion", "Cheese", "Cheese_sliced", "Dough", \
	"Beef", "Chicken", "BreadBP", "Lemon", "Mango", "Kiwi", "Peach", "Apple", "Orange", "Ham", "Turkey", \
	"Salami", "Potato"]
ContList = ["Fridge", "Plate", "Plate2", "Hand", "BreadBP", "Dough", "Pot", "Cup", "SauceBottle", "Board"]
CutableList = ["Lettuce", "Tomato", "Cucumber", "Eggplant", "Onion", "Potato", "Lemon", "Mango", "Kiwi", \
	"Peach", "Apple", "Orange"]
PeelableList = ["Apple", "Cucumber", "Kiwi", "Mango", "Orange", "Peach", "Potato"]
JuiceList = ["Cucumber", "Tomato", "Apple", "Kiwi", "Mango", "Orange", "Peach", "Lemon"]
GratableList = ["Cheese"]

env = DiscreteAgent({"Name":"Agent1", \
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

			"Head": {"Rot":{"Pitch":-45,"Yaw":0.0,"Roll":0}},
			"rgb": True, "depth": False, "mask": False,
			"scene": "2"
			}, task_type="PrepareDish"
)

def InitDict():
	for obj in IngredList:
		ObjDict[obj] = {"Pos": "Fridge", "Cut":False, "Peel":False, "Juice":False, "Cook":False}
	ObjDict['Sauce'] = {"Pos": "SauceBottle", "Cut":False, "Peel":False, "Juice":False, "Cook":False}
	ObjDict['Actor'] = {"Pos": "Orig"}
	ObjDict['Fridge'] = {"State": "Closed"}


def GoTo(loc, folder_name="", count=[0], f_label="", f_fluent=""):
	if loc not in PosList or loc not in env.data['objects_vis']:
		# print "Location not in list"
		pass

	else:
		try:
			a = env.GoToPos(loc)
			ObjDict['Actor']["Pos"] = loc

		except Exception as e:
			# print(e)
			pass

	a = env.pad()

	if folder_name:
		count[0] += 1
		rgb_image = Image.fromarray(a['rgb'])
		rgb_image.save(folder_name+"/"+str(count[0])+".png")

	if f_label:
		f_label.write("Goto ")
		f_label.write(loc)
		f_label.write("\n")

	if f_fluent:
		f_fluent.write(str(ObjDict))
		f_fluent.write("\n")
	return a

def Take(obj, folder_name="", count=[0], f_label="", f_fluent=""):
	# TODO: add constraint that one hand can take things at a time
	if obj not in env.data['objects']:
		# # print "not near the object",obj,", fail to take"
		pass

	if obj in LeftHandList:
		hand = "LeftHand"
	else:
		hand = "RightHand"

	if obj in CenterList:
		# if in center list, move to center first
		comp = "Center"
	elif obj in StaticMeshList:
		comp = "StaticMeshComponent0"
	else:
		comp = "ProcMesh"

	try:
		a = env.MoveToObject(hand, obj, comp)

		if obj in StaticMeshList:
			comp = "StaticMeshComponent0"
		else:
			comp = "ProcMesh"

		a = env.GrabObject(hand, obj, comp)
		a = env.MoveToNeutral(hand)

		ObjDict[obj]['Pos'] = "Hand"


	except Exception as e:
		# # print(e)
		pass

	a = env.pad()

	if folder_name:
		count[0] += 1
		rgb_image = Image.fromarray(a['rgb'])
		rgb_image.save(folder_name+"/"+str(count[0])+".png")

	if f_label:
		f_label.write("Take ")
		f_label.write(obj)
		f_label.write("\n")

	if f_fluent:
		f_fluent.write(str(ObjDict))
		f_fluent.write("\n")

	return a

def PlaceTo(obj, folder_name="", count=[0], f_label="", f_fluent=""):

	if env.state["LeftHand"]["ActorName"]:
		hand = "LeftHand"
		actor_in_hand = env.state["LeftHand"]["ActorName"]
		comp_in_hand = env.state["LeftHand"]["CompName"]
	elif env.state["RightHand"]["ActorName"]:
		hand = "RightHand"
		actor_in_hand = env.state["RightHand"]["ActorName"]
		comp_in_hand = env.state["RightHand"]["CompName"]
	else:
		hand = ""
		actor_in_hand = ""
		comp_in_hand = ""
		# print "nothing to place"

	if actor_in_hand in CenterList:
		# if in center list, move to center first
		comp = "Center"
	elif actor_in_hand in StaticMeshList:
		comp = "StaticMeshComponent0"
	else:
		comp = "ProcMesh"

	if obj in BoxTwoList:
		comp_place = "Box2"
	elif obj in BoxList:
		comp_place = "Box"
	elif obj in FoodPointList:
		comp_place = "FoodPoint"
	elif obj in CenterList:
		comp_place = "Center"
	elif obj in ContainerMesh:
		comp_place = "ContainerMesh"
	else:
		comp_place = "StaticMeshComponent0"

	try:
		a = env.MoveContactToObject(hand, comp, obj, comp_place)
		a = env.ReleaseObject(hand)
		a = env.MoveToNeutral(hand)

		if obj in ContList:
			ObjDict[actor_in_hand]["Pos"] = obj 

	except Exception as e:
		# print(e)
		pass

	a = env.pad()

	if folder_name:
		count[0] += 1
		rgb_image = Image.fromarray(a['rgb'])
		rgb_image.save(folder_name+"/"+str(count[0])+".png")

	if f_label:
		f_label.write("Placeto ")
		f_label.write(obj)
		f_label.write("\n")

	if f_fluent:
		f_fluent.write(str(ObjDict))
		f_fluent.write("\n")
	return a

def Use(tool, folder_name="", count=[0], f_label="", f_fluent=""):
	if env.state["LeftHand"]["ActorName"]:
		hand = "LeftHand"
		actor_in_hand = env.state["LeftHand"]["ActorName"]
		comp_in_hand = env.state["LeftHand"]["CompName"]
	elif env.state["RightHand"]["ActorName"]:
		hand = "RightHand"
		actor_in_hand = env.state["RightHand"]["ActorName"]
		comp_in_hand = env.state["RightHand"]["CompName"]
	else:
		hand = ""
		actor_in_hand = ""
		comp_in_hand = ""

	if not tool == "Oven" and tool not in env.data["objects"]:
		# print "not near the tool", tool
		pass

	try:
		if tool == "Knife":
			if not hand:
				pass
			else:
				a = env.MoveContactToObject(hand, "ProcMesh", "CutBoard", "Box")
				a = env.ReleaseObject(hand)
				a = env.MoveToNeutral(hand) 
				a = env.MoveToObject("RightHand", "Knife", "GrabPoint")
				a = env.GrabObject("RightHand", "Knife", "StaticMeshComponent0")
				a = env.MoveToNeutral("RightHand")
				a = env.step("RightHandTwistLeft", scale=5)
				a = env.step("RightHandMoveLeft", scale=2.7)
				a = env.MoveContactToObject("RightHand", "CutPoint", actor_in_hand, "ProcMesh")
				a = env.step("RightHandMoveUp", scale=2)
				a = env.MoveToNeutral("RightHand")
				a = env.step("RightHandTwistRight", scale=5)
				a = env.MoveContactToObject("RightHand", "GrabPoint", "282Main_18", "Box")
				a = env.ReleaseObject("RightHand")
				a = env.MoveToNeutral("RightHand")
				ObjDict[actor_in_hand]["Pos"] = "Board"
				if actor_in_hand in CutableList:
					ObjDict[actor_in_hand]["Cut"] = True

		elif tool == "Peeler":
			if hand == "LeftHand" or not hand:
				# print "LeftHand not empty"
				pass
			else:
				a = env.MoveToObject("LeftHand", "Peeler", "grabpoint")
				a = env.GrabObject("LeftHand", "Peeler", "StaticMeshComponent0")
				a = env.MoveToNeutral("LeftHand")
				a = env.MoveContactToObject("LeftHand","Blade", actor_in_hand, "Skin")
				a = env.MoveToNeutral("LeftHand")
				a = env.MoveContactToObject("LeftHand", "grabpoint", "282Main3", "Box")
				a = env.ReleaseObject("LeftHand")
				a = env.MoveToNeutral("LeftHand")
				if actor_in_hand in PeelableList:
					ObjDict[actor_in_hand]["Peel"] = True

		elif tool == "Grater":
			if hand == "RightHand" or not hand:
				# print "RightHand not empty"
				pass
			else:
				a = env.MoveToObject("RightHand", "Grater", "grabpoint")
				a = env.GrabObject("RightHand", "Grater", "StaticMeshComponent0")
				a = env.MoveToNeutral("RightHand")
				a = env.MoveContactToObject("RightHand", "StaticMeshComponent0", actor_in_hand, "ProcMesh")
				a = env.MoveToNeutral("RightHand")
				a = env.MoveContactToObject("RightHand", "StaticMeshComponent0", "284Main_2", "Box")
				a = env.ReleaseObject("RightHand")
				a = env.MoveToNeutral("RightHand")
				a = env.MoveContactToObject("LeftHand", "ProcMesh", "284Main_2", "Box1")
				a = env.ReleaseObject("LeftHand")
				a = env.MoveToNeutral("LeftHand")
				if actor_in_hand in GratableList:
					for thing in ObjDict:
						if thing in ContList and ObjDict[thing]['Pos'] == "Plate":
							ObjDict[actor_in_hand]["Pos"] = thing
							break

		elif tool == "SauceBottle":
			if hand == "RightHand":
				# print "RightHand not empty"
				pass
			else:
				a = env.MoveToObject("RightHand", "SauceBottle", "StaticMeshComponent0")
				a = env.GrabObject("RightHand","SauceBottle","StaticMeshComponent0")
				a = env.MoveToNeutral("RightHand")
				a = env.MoveContactToObject("RightHand","StaticMeshComponent0", "Plate2","Box2")
				a = env.step("RightHandTwistLeft", scale=6)
				a = env.step("RightHandTwistRight", scale=6)
				a = env.MoveToNeutral("RightHand")
				a = env.MoveContactToObject("RightHand","StaticMeshComponent0", "281Main_5","Box")
				a = env.ReleaseObject("RightHand")
				a = env.MoveToNeutral("RightHand")
				for thing in ObjDict:
					if thing in ContList and ObjDict[thing]['Pos'] == "Plate2":
						ObjDict["Sauce"]["Pos"] = thing
						break

		elif tool == "Juicer":
			if not hand:
				# print "nothing to juice"
				pass
			else:
				a = env.GoToPos("Juicer")
				a = env.MoveContactToObject(hand,"ProcMesh", "Juicer", "SqueezeBox")
				a = env.MoveToNeutral(hand)
				if actor_in_hand in JuiceList:
					ObjDict[actor_in_hand]["Pos"] = "Cup"
					ObjDict[actor_in_hand]["Juice"] = True

		elif tool == "Cup":
			if hand == "LeftHand":
				# print "LeftHand not empty"
				pass
			else:
				a = env.MoveToObject("LeftHand", "Cup", "GrabPoint")
				a = env.GrabObject("LeftHand", "Cup", "ContainerMesh")
				a = env.MoveToNeutral("LeftHand")
				a = env.GoToPos("Stove")
				a = env.MoveContactToObject("LeftHand","ContainerMesh", "Pot","PlaceForBlender")
				a = env.step("LeftHandTwistRight", scale=5)
				a = env.step("LeftHandTwistLeft", scale=5)
				a = env.MoveToNeutral("LeftHand")
				a = env.GoToPos("Juicer")
				a = env.MoveContactToObject("LeftHand","ContainerMesh", "Juicer","PlaceForCup")
				a = env.ReleaseObject("LeftHand")
				a = env.MoveToNeutral("LeftHand")
				for thing in ObjDict:
					if ObjDict[thing]['Pos'] == "Cup":
						ObjDict[thing]['Pos'] = "Pot"

		elif tool == "Oven":
			if not hand:
				# print "nothing to place into oven"
				pass
			else:
				if hand == "LeftHand":
					open_stove_hand = "RightHand"
				else:
					open_stove_hand = "LeftHand"
				a = env.Crouch()
				a = env.MoveToObject(open_stove_hand, "StoveDoor", "Box")
				a = env.MoveToNeutral(open_stove_hand)
				a = PlaceTo("Stove")
				a = Take(actor_in_hand)
				a = env.Standup()
				a = env.Standup()
				for thing in ObjDict:
					if ObjDict[thing]['Pos'] == "Hand" or ObjDict[thing]['Pos'] == actor_in_hand:
						ObjDict[thing]['Cook'] = True
	except Exception as e:
		# print(e)
		pass

	a = env.pad()

	if folder_name:
		count[0] += 1
		rgb_image = Image.fromarray(a['rgb'])
		rgb_image.save(folder_name+"/"+str(count[0])+".png")

	if f_label:
		f_label.write("Use ")
		f_label.write(tool)
		f_label.write("\n")

	if f_fluent:
		f_fluent.write(str(ObjDict))
		f_fluent.write("\n")

	return a

def Open(tool, folder_name="", count=[0], f_label="", f_fluent=""):
	if env.state["LeftHand"]["ActorName"]:
		hand = "LeftHand"
		actor_in_hand = env.state["LeftHand"]["ActorName"]
		comp_in_hand = env.state["LeftHand"]["CompName"]
	elif env.state["RightHand"]["ActorName"]:
		hand = "RightHand"
		actor_in_hand = env.state["RightHand"]["ActorName"]
		comp_in_hand = env.state["RightHand"]["CompName"]
	else:
		hand = ""
		actor_in_hand = ""
		comp_in_hand = ""

	try:
		if tool == "Fridge":
			a = env.MoveToObject("RightHand", "Fridge", "Box")
			a = env.MoveToNeutral("RightHand")
			ObjDict[tool]["State"] = "Open"
		elif tool == "Stove":
			a = env.MoveToObject("LeftHand","Stove","Switch")
			a = env.MoveToObject("LeftHand","Stove","Switch")
			a = env.MoveToNeutral("LeftHand")
			ObjDict[tool]["State"] = "Open"
			for thing in ObjDict:
				if ObjDict[thing]['Pos'] == "Pot":
					ObjDict[thing]['Cook'] = True

	except Exception as e:
		# print(e)
		pass

	a = env.pad()

	if folder_name:
		count[0] += 1
		rgb_image = Image.fromarray(a['rgb'])
		rgb_image.save(folder_name+"/"+str(count[0])+".png")

	if f_label:
		f_label.write("Open ")
		f_label.write(tool)
		f_label.write("\n")

	if f_fluent:
		f_fluent.write(str(ObjDict))
		f_fluent.write("\n")

	return a



ObjDict = {}
InitDict()
# a = env.start()











