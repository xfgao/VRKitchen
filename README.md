# VRKitchen #

VRKitchen is an interactive 3D virtual kitchen environment which provides a platform for training and evaluating various learning and planning algorithms in a variety of cooking tasks. Two kinds of tasks are available:

1. Tool Use: requires an agent to continuously control its hands to make use of a tool.
2. Preparing dishes: agents must perform a series of atomic actions in the right order to achieve a compositional goal.

## Requirements ##
* OS: Ubuntu 16.04
* Python 2.7.12
* numpy==1.16.2
* torch==0.4.1
* pillow==5.4.1
* scikit-image==0.14.2
* torchvision==0.2.1
* psutil==3.4.2
* opencv-python==4.0.0.21
* tqdm==4.31.1
* pathlib2==2.3.3
* pyrapidjson==0.5.1
* tensorboardX==1.7
* gym==0.12.5

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
