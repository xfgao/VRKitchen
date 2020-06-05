from recipe_api import GoTo, Take, PlaceTo, Use, Open, env, ObjDict, InitDict
from PIL import Image
import os
import sys
import random
retry_num = 3
data_num = 20
random.seed(0)

# make juice
scenes = ["2", "3", "5", "6"]
fruits = ["Lemon", "Kiwi", "Orange", "Apple", "Peach", "Mango"]
for i in range(data_num):
    idx1 = random.randint(0, 5)
    fruit = fruits[idx1]
    idx2 = random.randint(0, 5)
    fruit2 = fruits[idx2]
    if idx1 == idx2:
        continue
    idx3 = random.randint(0, 3)
    scene = scenes[idx3]
    folder_name = "dataset/make_juice/"+fruit+"_"+fruit2+"_"+"scene"+scene
    try:
        os.mkdir(folder_name)
    except:
        print "folder exists"
    try_times = 0
    while try_times<retry_num:
        try:
            f_label = open(folder_name+"/action_label.txt", "w")
            f_fluent = open(folder_name+"/fluent.txt", "w")
            InitDict()
            count = [0]
            a = env.reset(scene)
            env.state['rgb'] = True
            a = env.send_tf()
            env.state['rgb'] = False
            rgb_image = Image.fromarray(a['rgb'])
            rgb_image.save(folder_name+"/"+str(count[0])+".jpg")

            f_label.write("None")
            f_label.write("\n")
            f_fluent.write(str(ObjDict))
            f_fluent.write("\n")

            a = GoTo("Fridge",folder_name, count, f_label, f_fluent)
            a = Open("Fridge",folder_name, count, f_label, f_fluent)
            a = Take(fruit,folder_name, count, f_label, f_fluent)
            a = GoTo("Knife",folder_name, count, f_label, f_fluent)
            a = Use("Knife",folder_name, count, f_label, f_fluent)
            a = Take(fruit,folder_name, count, f_label, f_fluent)
            a = GoTo("Juicer",folder_name, count, f_label, f_fluent)
            a = Use("Juicer",folder_name, count, f_label, f_fluent)
            a = GoTo("Fridge",folder_name, count, f_label, f_fluent)
            a = Take(fruit2,folder_name, count, f_label, f_fluent)
            a = GoTo("Knife",folder_name, count, f_label, f_fluent)
            a = Use("Knife",folder_name, count, f_label, f_fluent)
            a = Take(fruit2,folder_name, count, f_label, f_fluent)
            a = GoTo("Juicer",folder_name, count, f_label, f_fluent)
            a = Use("Juicer",folder_name, count, f_label, f_fluent)
            f_label.close()
            f_fluent.close()
            break
        except Exception as e:
            print e
            print "retry"
        try_times += 1
    if try_times == retry_num:
        print folder_name