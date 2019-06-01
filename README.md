# VRKitchen #

VRKitchen is an interactive 3D virtual kitchen environment which provides a platform for training and evaluating various learning and planning algorithms in a variety of cooking tasks. Two kinds of tasks are available:

1. Tool Use: requires an agent to continuously control its hands to make use of a tool.
  E.g. cutting a carrot into pieces, peeling a kiwi, pouring water from one cup to another. 

2. Preparing dishes: agents must perform a series of atomic actions in the right order to achieve a compositional goal.
  E.g. making fruit juice, beef stew and sandwiches. 

## Requirements ##
* OS: Ubuntu 16.04
* Python 2.7.12

## Getting Started ##
1. Clone the repo and install the requirements.

```bash
git clone https://github.com/xfgao/VRKitchen.git
pip install -r requirements.txt
```

2. Download the zip file and unzip it into the /Binaries folder.

- [Download] https://github.com/xfgao/VRKitchen/releases/download/v0.1.0/VRKitchen-v0.1.0.zip

3. To reproduce the experiments, run the Script/example_dish.py (for preparing dishes) and Script/example_tool.py (for tool use)

```bash
python Script/example_dish.py
python Script/example_tool.py
```

  Have fun!
