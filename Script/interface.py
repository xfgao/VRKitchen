from recipe_api import recipe 
from DiscreteAgent import DiscreteAgent
from tool_pos import tool_pos

from flask import Flask,render_template,request, make_response
import os
import numpy as np
import random
import copy
import time
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pymongo import MongoClient
import uuid
import json
import shutil

import time
import unicodedata
r = recipe()

pos = {"Orig":"","Juicer":"","Knife":"Board","Oven":"","Peeler":"",
"Sauce": "Plate2", "Grater":"Plate", "Stove":"Pot", "Fridge":"Fridge"}

app = Flask(__name__, static_url_path='/static')
app.jinja_env.filters['zip'] = zip
tool = tool_pos["2"]
active = False  # active means user chose the task and scene id and hit the start button
fridgeOpen = False # if fridge is open
task = "" # task user chooses
count = [-1] 
folder_name = "" # folder to save images
f_label = None # file descriptor to save action label
f_fluent = None # file descriptor to save fluent change
level = None # lable of user's choice
loc = None  # location of the robot
done = False # done means that user successfully finish the task
ingredients  = [] # ingredients needed in the cooking process
goal_state = [] # the state we want robot to acheive, fluent, location etc
ingred1 = None # first ingredient
ingred2 = None # second ingredient
uid = None# user id

@app.route('/level',methods=['GET', 'POST'])
def changeLevel():
	global level, active
	form = request.form
	if not active: 
		level = form['level']
	return index()

@app.route('/start',methods=['GET', 'POST'])
def start():
	global r,active,fridgeOpen,task,count,folder_name,level,tool,loc
	global f_label, f_fluent, goal_state, ingredients, done, uid
	if active == True:
		return index()
	if uid == None:
		uid = uuid.uuid1()
	done = False
	r.ObjDict = {}
	r.InitDict()
	if not task:
		return index()
	if active == False:
		try:
			r.env.reset(level)
		except:
			pass
		active = True
		fridgeOpen = False
		tool = tool_pos[level]
	if "fridgeDoorDown" not in r.OpenList:
		r.OpenList.append("fridgeDoorDown")


	loc = "Orig"
	count = [-1]
	folder_name = ""

	
	if task == "peel fruit":
		fruit = np.random.choice(r.PeelableList)
		folder_name = "dataset/peel_fruit/"+fruit+"_"+"scene"+str(level)+"_"+str(uid)
		ingredients = [fruit]
		goal_state = sorted([(fruit, {'Pos': 'Hand', 'Peel': 'True'})])

	elif task == "cut fruit":
		fruit = np.random.choice(r.CutableList)
		folder_name = "dataset/cut_fruit/"+fruit+"_"+"scene"+str(level)+"_"+str(uid)
		ingredients = [fruit]
		goal_state = sorted([(fruit, {'Cut': 'True', 'Pos': 'Board'})])

	elif task == "make juice":
		fruits = ["Lemon", "Kiwi", "Orange", "Apple", "Peach", "Mango"]
		temp =  np.random.choice(fruits,2, replace=False).tolist()
		folder_name = "dataset/make_juice/"+temp[0]+"_"+temp[1]+"_"+"scene"+str(level)+"_"+str(uid)
		ingredients = temp
		goal_state = [(temp[0], {'Juice': 'True', 'Cut': 'True', 'Pos': 'Cup'}), 
				(temp[1], {'Juice': 'True', 'Cut': 'True', 'Pos': 'Cup'})]

	elif task == "cook meat":
		fruits = ["Apple", "Kiwi", "Mango", "Orange", "Peach", "Lemon"]
		meats = ["Beef", "Chicken"]
		temp1 = np.random.choice(fruits)
		temp2 = np.random.choice(meats)
		folder_name = "dataset/cook_meat/"+temp1+"_"+temp2+"_"+"scene"+str(level)+"_"+str(uid)
		ingredients = [temp1]
		ingredients.append(temp2)
		
		goal_state = sorted([(temp2, {'Cook': 'True', 'Pos': 'Pot'}), 
		(temp1, {'Cook': 'True', 'Juice': 'True', 'Cut': 'True', 'Pos': 'Pot'})])

	elif task == "cook soup":
		vegs = ["Cucumber", "Tomato", "Eggplant", "Onion"]
		meats = ["Beef", "Chicken"]
		temp1 = np.random.choice(vegs)
		temp2 = np.random.choice(meats)
		folder_name = "dataset/cook_soup/"+temp1+"_"+temp2+"_"+"scene"+str(level)+"_"+str(uid)
		ingredients = [temp1]
		ingredients.append(temp2)

		goal_state = sorted([(temp1, {'Cook': 'True', 'Cut': 'True', 'Pos': 'Pot'}), 
		 (temp2, {'Cook': 'True','Pos': 'Pot'})])

	elif task == "make sandwich":
		ingredients = ["BreadBP", 'Cheese_sliced', 'Sauce']

		vegs = ["Cucumber", "Tomato", "Eggplant", "Onion"]
		coldcuts = ["Salami", "Ham", "Turkey"]
		temp1 = np.random.choice(vegs)
		temp2 = np.random.choice(coldcuts)
		folder_name = "dataset/make_sandwich/"+temp1+"_"+temp2+"_"+"scene"+str(level)+"_"+str(uid)
		ingredients.append(temp1)
		ingredients.append(temp2)

		goal_state =  sorted([('BreadBP', {'Cook': 'True', 'Pos': 'Plate2'}), 
		('Cheese_sliced', {'Cook': 'True',  'Pos': 'BreadBP'}), 
		('Sauce', {'Pos': 'BreadBP'}), 
		(temp1, {'Cut': 'True', 'Pos': 'BreadBP'}), 
		(temp2, {'Cook': 'True', 'Pos': 'BreadBP'})])

	elif task == "make pizza":
		ingredients = ['Dough', 'Sauce','cheese']
		vegs = ["Cucumber", "Tomato", "Eggplant", "Onion"]
		coldcuts = ["Salami", "Ham", "Turkey"]
		temp1 = np.random.choice(vegs)
		temp2 = np.random.choice(coldcuts)
		folder_name = "dataset/make_pizza/"+temp1+"_"+temp2+"_"+"scene"+str(level)+"_"+str(uid)
		ingredients.append(temp1)
		ingredients.append(temp2)

		goal_state =  sorted([('Dough', {'Cook': 'True','Pos': 'Plate2'}), 
		('Sauce', {'Cook': 'True','Pos': 'Dough'}), 
		('cheese', {'Cook': 'True', 'Pos': 'Dough'}), 
		(temp1, {'Cook': 'True', 'Cut': 'True', 'Pos': 'Dough'}), 
		(temp2, {'Cook': 'True', 'Pos': 'Dough'})])

	try:
		os.makedirs(folder_name)
	except:
		print "folder exist"

	try:
		f_label = open(folder_name+"/action_label.txt", "w")
		f_fluent = open(folder_name+"/fluent.txt", "w")
	except:
		print "files already open"

	r.GoTo('Orig', count = count, 
			folder_name = folder_name, f_fluent= f_fluent)
	return index()

@app.route('/end',methods=['GET', 'POST'])
def end():
	global f_label, f_close, active,task,done, folder_name
	if f_label:
		f_label.close()
	if f_fluent:
		f_fluent.close()
	#if done == False and active:
		#open(folder_name+"/action_label.txt", "w").close()
		#open(folder_name+"/fluent.txt", "w").close()
		#shutil.rmtree(folder_name)
	done = False
	active = False
	task = ""
	r.ObjDict = {}
	r.InitDict()
	return index()

@app.route('/',methods=['GET', 'POST'])
def index():
	global r,f_label, f_fluent,active, fridgeOpen,task,count, level,tool,loc
	global ingredients, goal_state, done

	time.sleep(0.1)
	form = request.form
	vid = ""
	msg = ""
	
	
	if 'level' in form:
			level = form['level']
			msg = 'You chose level ' + level + """. Please choose a task or change level. 
			You cannot change level once you start the game"""
	elif not level:
			msg = "Please choose a scene id."
	elif 'task' in form and active:
			msg = "Please end and chose a new task. You cannot change task in the middle."
	elif 'task' in form:
				task = form['task'] 
				msg = "You chose " + task + """\n Please press start button to start or change task.
				 You cannot change task once it starts"""
				temp =  task.split()
				vid = temp[0]+"_"+temp[1]+".mp4"
	elif not task:
			msg = "Please choose a task first";
	elif active == False:
			msg = "Press start button to start"

	elif 'goto' in form:
			msg = r.GoTo(str(form['goto']), count = count, 
				folder_name = folder_name, f_fluent= f_fluent, f_label = f_label)
			if msg == "Success":
				loc = form['goto']
			

	elif 'take' in form:
			msg = r.Take(str(form['take']), count = count, 
				folder_name = folder_name, f_fluent= f_fluent, f_label = f_label)
			

	elif 'placeto' in form:
			msg = r.PlaceTo(str(form['placeto']), count = count, 
				folder_name = folder_name, f_fluent= f_fluent, f_label = f_label)
			

	elif 'use' in form:
			msg = r.Use(str(form['use']), count = count, 
				folder_name = folder_name, f_fluent= f_fluent, f_label = f_label)
			

	elif 'open' in form:
			if str(form['open']) == 'Fridge':
				msg = r.Open("Fridge", count = count, 
					folder_name = folder_name, f_fluent= f_fluent, f_label = f_label)
				fridgeOpen = True

			else:
				msg = r.Open(str(form['open']), count = count, 
					folder_name = folder_name, f_fluent= f_fluent, f_label = f_label)
	elif 'head' in form:
			if str(form['head']) == "look up":
				msg = r.LookUp(f_label=f_label)
			elif str(form['head']) == "look down":
				msg = r.LookDown(f_label=f_label)
			count = [count[0]-1]
			msg = r.GoTo(loc, count = count, 
				folder_name = folder_name)


			
	time.sleep(0.1)
	#################### goto
	goto = tool.keys()

	near = set(r.env.data['objects'])

	#####################   take
	temp = set()
	if fridgeOpen:
		temp = set(r.IngredList)

	
	for key, value in r.ObjDict.iteritems():
		if loc  and str(value["Pos"])  != str(pos[loc]) and key in temp:
			temp.remove(key)


	#cannot take it away while on bread or pizza
	for key, value in r.ObjDict.iteritems():
		if (value["Pos"] == "BreadBP" or value["Pos"] == "Dough") and key in temp :
			temp.remove(key)
		

	# cannot take it while are on the hand
	take = list(near.intersection(temp))
	if r.env.state["LeftHand"]["ActorName"] in take:
		take.remove(r.env.state["LeftHand"]["ActorName"])
	if r.env.state["RightHand"]["ActorName"] in take:
		take.remove(r.env.state["RightHand"]["ActorName"])

	# cannot take it while hands are not empty
	if r.env.state["RightHand"]["ActorName"] or r.env.state["LeftHand"]["ActorName"]:
		take = []

	

	##################### place to
	temp = set()

	if "Fridge" in tool and r.env.state["Actor"]["Loc"] == tool["Fridge"]["Actor"]["Loc"] and (r.env.state["LeftHand"]["ActorName"]  or 
 		r.env.state["RightHand"]["ActorName"]):
		if r.env.state["LeftHand"]["ActorName"] : 
			obj = r.env.state["LeftHand"]["ActorName"]
		else:
			obj = r.env.state["RightHand"]["ActorName"]
		if obj == "BreadBP" or obj == "Cheese_sliced" or obj == "Dough":
			pass
		else:
 			temp.add("Fridge")

	if "Sauce" in tool and  r.env.state["Actor"]["Loc"] == tool["Sauce"]["Actor"]["Loc"] and (r.env.state["LeftHand"]["ActorName"]  or 
		r.env.state["RightHand"]["ActorName"]):
		temp.add("Plate2")


	if "Grater" in tool and (r.env.state["Actor"]["Loc"] == tool["Grater"]["Actor"]["Loc"] and
		(r.env.state["LeftHand"]["ActorName"]  or 
		r.env.state["RightHand"]["ActorName"]  )):
		temp.add("Plate")
	if "Stove" in tool and (r.env.state["Actor"]["Loc"] == tool["Stove"]["Actor"]["Loc"] and
		(r.env.state["LeftHand"]["ActorName"]  or 
		r.env.state["RightHand"]["ActorName"]  )):
		temp.add("Pot")

	## interate obj list to find position of Dough and BreadBP 
	
	for key, value in r.ObjDict.iteritems():
			locActor = r.env.state["Actor"]["Loc"]
			
			if key == "Dough":
				if  ( ( str(value["Pos"]) == "Plate" and   \
				locActor  == tool[ "Grater" ]["Actor"]["Loc"])) and   \
				(r.env.state["LeftHand"]["ActorName"]  or r.env.state["RightHand"]["ActorName"]  ): 
					temp.add("Dough")
					temp.remove("Plate")
				if  ( ( str(value["Pos"]) == "Plate2" and   \
				locActor  == tool["Sauce"]["Actor"]["Loc"])) and   \
				(r.env.state["LeftHand"]["ActorName"]  or r.env.state["RightHand"]["ActorName"]  ): 
					temp.add("Dough")
					temp.remove("Plate2")

				
			elif key == "BreadBP":
				if  ( ( str(value["Pos"]) == "Plate" and   \
				locActor  == tool[ "Grater" ]["Actor"]["Loc"])) and   \
				(r.env.state["LeftHand"]["ActorName"]  or r.env.state["RightHand"]["ActorName"]  ): 
					temp.add("BreadBP")
					temp.remove("Plate")
				if  ( ( str(value["Pos"]) == "Plate2" and   \
				locActor  == tool["Sauce"]["Actor"]["Loc"])) and   \
				(r.env.state["LeftHand"]["ActorName"]  or r.env.state["RightHand"]["ActorName"]  ): 
					temp.add("BreadBP")
					temp.remove("Plate2")
		
		

	#### issue here for staticmeshcomponent0
	#placeto = list(near.intersection(temp))
	placeto = temp
	
	
	if not r.env.state["LeftHand"]["ActorName"] and not r.env.state["RightHand"]["ActorName"]:
		placeto = []
	if r.env.state["LeftHand"]["ActorName"] in placeto:
		placeto.remove(r.env.state["LeftHand"]["ActorName"])
	if r.env.state["RightHand"]["ActorName"] in placeto:
		placeto.remove(r.env.state["RightHand"]["ActorName"])

	if  ((r.env.state["LeftHand"]["ActorName"] in r.IngredList or 
		r.env.state["RightHand"]["ActorName"] in r.IngredList) and 
		"Cup" in placeto):
		placeto.remove("Cup")
	#print placeto
	#################  use tools

	temp = set()
	

	if "Sauce" in tool and  r.env.state["Actor"]["Loc"] == tool["Sauce"]["Actor"]["Loc"]:
		temp.add("SauceBottle")


	if "Knife" in tool and  (r.env.state["Actor"]["Loc"] == tool["Knife"]["Actor"]["Loc"] and
		(r.env.state["LeftHand"]["ActorName"] in r.CutableList or 
		r.env.state["RightHand"]["ActorName"] in r.CutableList)):
		temp.add("Knife")

	if "Peeler" in tool and (r.env.state["Actor"]["Loc"] == tool["Peeler"]["Actor"]["Loc"] and
		(r.env.state["LeftHand"]["ActorName"] in r.PeelableList or 
		r.env.state["RightHand"]["ActorName"] in r.PeelableList)):
		temp.add("Peeler")

	if "Juicer" in tool and (r.env.state["Actor"]["Loc"] == tool["Juicer"]["Actor"]["Loc"] and
		(r.env.state["LeftHand"]["ActorName"] in r.JuiceList or 
		r.env.state["RightHand"]["ActorName"] in r.JuiceList)):
		temp.add("Juicer")

	if "Juicer" in tool and (r.env.state["Actor"]["Loc"] == tool["Juicer"]["Actor"]["Loc"]):
		temp.add("Cup")		


	if "Grater" in tool and (r.env.state["Actor"]["Loc"] == tool["Grater"]["Actor"]["Loc"] and
		(r.env.state["LeftHand"]["ActorName"] in r.GratableList  or 
		r.env.state["RightHand"]["ActorName"] in r.GratableList )):
		temp.add("grater")

	if (r.env.state["Actor"]["Loc"] == tool["Oven"]["Actor"]["Loc"]):
		temp.add("Oven")


	use = list(temp)
	############## open	
	temp = set()
	if r.env.state["Actor"]["Loc"] == tool["Fridge"]["Actor"]["Loc"] and fridgeOpen == False :
		temp.add("Fridge")

	if r.env.state["Actor"]["Loc"] == tool["Stove"]["Actor"]["Loc"]:
		temp.add("Stove")

	openn = list(temp)
	########## compute current state
	state = []
	
	for obj in ingredients:
		try:
			temp = dict()
			for key, value in r.ObjDict[obj].iteritems():
				if str(value) == "True" or key == "Pos":
					temp[key] = str(value)
			state.append((obj,temp))
		except:
			pass



	if cmp(sorted(state), sorted(goal_state) )==0 and active == True:
		done = True	
		r.RecordSuccess(f_label = f_label)

	if msg == "Success":
		msg = None
	return render_template("index.html", goto=sorted(goto), take = sorted(take), placeto = sorted(placeto), use=sorted(use), 
		open = sorted(openn), msg = msg, ingredients = sorted(ingredients), level = level, 
		task = task, active = active, loc = loc, state= sorted(state), goal = goal_state, done = done,vid=vid )
	#return render_template("index.html", goto=goto, take = list(near), placeto = list(near), use=list(near), open = list(near), msg = msg)



if __name__ == '__main__':
	app.run(host='127.0.0.2', port = 9000)
	