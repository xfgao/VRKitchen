from recipe_api import recipe 
from DiscreteAgent import DiscreteAgent
import time

r = recipe()
filename = open('action_label.txt', 'r') 
Lines = filename.readlines() 

for line in Lines:
    print line
    commands = line.rstrip().split(' ')
    if commands[0].lower() == "goto":
        r.GoTo(commands[1])
    elif commands[0].lower() == "take":
        r.Take(commands[1])
    elif commands[0].lower() == "use":
        r.Use(commands[1])
    elif commands[0].lower() == "placeto":
        r.PlaceTo(commands[1])
    elif commands[0].lower() == "open":
        r.Open(commands[1])
    elif commands[0].lower() == "look":
        if commands[1].lower() == "up":
            r.LookUp()
        else:
            r.LookDown()
    
