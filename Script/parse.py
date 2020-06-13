from recipe_api import recipe 
from DiscreteAgent import DiscreteAgent
from PIL import Image

import time
import os

dir = 'person1'
mid_dirs = os.listdir(dir)
full_path = []
for d in mid_dirs:
    mid_path = os.path.join(dir, d)
    low_dirs = os.listdir(mid_path)
    full_path.extend([os.path.join(mid_path, key) for key in low_dirs  ])



r = recipe()

for f in ['person1/cook_soup/Tomato_Beef_scene3_260b7924-d7fb-11e8-8272-94b86dd32278']:
    level = str(f[f.find('scene')+5])
    r.env.reset(level)
    filename = open(os.path.join(f, 'action_label.txt'), 'r') 
    Lines = filename.readlines() 
    print f
    # try:
    count = 0
    try:
        while True:
        # for line in Lines:
            # print line, count
            # commands = line.rstrip().split(' ')
            # if commands[0].lower() == "goto":
            #     r.GoTo(commands[1])
            # elif commands[0].lower() == "take":
            #     r.Take(commands[1])
            # elif commands[0].lower() == "use":
            #     r.Use(commands[1])
            # elif commands[0].lower() == "placeto":
            #     r.PlaceTo(commands[1])
            # elif commands[0].lower() == "open":
            #     r.Open(commands[1])
            # elif commands[0].lower() == "look":
            #     if commands[1].lower() == "up":
            #         r.LookUp()
            #     else:
            #         r.LookDown()
            # else:
            #     raise Exception("undefined command", command)
            
  
            while True:
                try:
                    r.env.state['rgb'] = True
                    res = r.env.NoOp()
                    rgb_image = Image.fromarray(res['rgb'])
                    break
                except:
                    pass
                time.sleep(0.1)

            r.env.state['rgb'] = False
            r.env.NoOp()

        # except:
        #     print "error in {%s}".format(f) 
            # rgb_image.save(f+"/"+str(count)+".jpg")
            # count += 1
    except:
        pass
        
