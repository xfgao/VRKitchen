from recipe_api import recipe
from tool_pos import tool_pos
r = recipe()
grab = ["Kiwi", "Eggplant", "Apple", "Lemon", "Orange", "BreadBP", "Dough", "Lettuce", "Salami", "Chicken", "Beef", "Onion", "Cucumber", "Potato", "Cheese_sliced", "Cheese", "Tomato"]
cnt = 0
count = None
take = None
prevCount = None
loc = None
plate = []
plate2 = []
pot = []
board = []
while True:
    a = r.env.NoOp()
    command =  a["command"]
    try:
        prevCount = count
        count = int(a["count"])
    except:
        continue
    #print command
    if prevCount == count:
        #print prevCount
        #print count
        continue


    if command in tool_pos[r.env.state["scene"]] and cnt == count:
        if tool_pos[r.env.state["scene"]][command]["Actor"] != r.env.state["Actor"]:
            res = r.GoTo(command)
            loc = command
        elif command == "Fridge":
            res = r.Open(command)
        elif command == "Stove":
            res = r.Open(command)
        elif command == "Juicer" and take != None:
            res = r.Use("Juicer")
            take = None

        elif command == "Knife" and take != None:
            res = r.Use("Knife")
            board.append(take)
            take = None
           #res = r.Take(take)
        elif command == "Sauce":
          res = r.Use("SauceBottle")
        elif command == "Oven":
          res = r.Use("Oven")
        elif command == "Grater" and take == "Cheese":
          res = r.Use("Grater")
          take = None
            
    elif "BreadBP" == command and ((loc == "Grater" and "BreadBP" in plate and take != None) or 
								   (loc == "Sauce" and "BreadBP" in plate2 and take != None)  ):
        r.PlaceTo(command)
        take = None
    elif "BreadBP" == command and  ((loc == "Grater" and "BreadBP" in plate and take == None) or 
								   (loc == "Sauce" and "BreadBP" in plate2 and take == None)  ):
        if "BreadBP" in plate:
          plate.remove("BreadBP")

        if "BreadBP" in plate2:
          plate2.remove("BreadBP")
        take = "BreadBP"
        r.Take("BreadBP")

    elif "Dough" == command and ((loc == "Grater" and "Dough" in plate and take != None) or 
								   (loc == "Sauce" and "Dough" in plate2 and take != None)  ):
        r.PlaceTo(command)
        take = None
    elif "Dough" == command and  ((loc == "Grater" and "Dough" in plate and take == None) or 
								   (loc == "Sauce" and "Dough" in plate2 and take == None)  ):
        if "Dough" in plate:
          plate.remove("Dough")
        if "Dough" in plate2: 
          plate2.remove("Dough")
        take = "Dough"
        r.Take("Dough")

    elif "Stove" == loc and command == "Pot" and cnt == count:
        res = r.PlaceTo("Pot")
        take = None    
    elif "Grater" == loc and command == "Plate" and cnt == count:
        res = r.PlaceTo("Plate")
        plate.append(take)
        take = None
      
    elif "Sauce" == loc and command == "Plate2" and cnt == count:
        res = r.PlaceTo("Plate2")
        plate2.append(take)
        
        take = None
    elif loc == "Knife" and command == "Board" and cnt == count and take == None and board:
        take = board[0]
        r.Take(take)
        
    elif  command in grab and cnt == count and take == None and (loc == "Plate" and command in plate) :
        res = r.Take(command)
        take = command
        plate.remove(take)
    elif  command in grab and cnt == count and take == None and  (loc == "Plate2" and command in plate2):
        res = r.Take(command)
        take = command
        plate2.remove(take)

    elif  command in grab and cnt == count and take == None and  (loc == "Stove" and command in pot ):
        res = r.Take(command)
        take = command
        pot.remove(take)
    elif  command in grab and cnt == count and take == None and  (loc == "Knife" and command in board ):
        res = r.Take(command)
        take = command
        board.remove(take)
    elif  command in grab and cnt == count and take == None and  (loc == "Fridge" ):
        res = r.Take(command)
        take = command
        
      
    elif "Juicer" == loc and command == "Cup" and cnt == count:
        res = r.Use(command)

    print count
    print cnt
    print command
    print board
    cnt += 1       




  #tool_pos[r.env.state["scene"]]["Fridge"]["Actor"] == r.env.state["Actor"] and