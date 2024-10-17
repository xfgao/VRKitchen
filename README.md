# VRKitchen #

VRKitchen is an interactive 3D virtual kitchen environment which provides a platform for training and evaluating various learning and planning algorithms in a variety of cooking tasks. Two kinds of tasks are available:

1. **Tool Use**: requires an agent to continuously control its hands to make use of a tool.

    E.g. cutting a carrot into pieces, peeling a kiwi, pouring water from one cup to another. 

2. **Preparing dishes**: agents must perform a series of atomic actions in the right order to achieve a compositional goal.

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

3. To reproduce the experiment results, run the Script/example_dish.py (for preparing dishes) and Script/example_tool.py (for tool use).

```bash
python Script/example_dish.py
python Script/example_tool.py
```

  Have fun!

## Dataset ##
Dataset for the dish preparation task: [https://drive.google.com/file/d/1QTAvO6Uwm_paGAfGMveKHCMxwAxGAyi_/view?usp=sharing](https://drive.google.com/file/d/1yahh4RGl7ofepoJin57n2mV1g-i7-T0L/view?usp=sharing)

## Citation ##
```bash
@article{VRKitchen,
  author    = {Xiaofeng Gao and
               Ran Gong and
               Tianmin Shu and
               Xu Xie and
               Shu Wang and
               Song{-}Chun Zhu},
  title     = {VRKitchen: an Interactive 3D Virtual Environment for Task-oriented
               Learning},
  journal   = {arXiv},
  volume    = {abs/1903.05757},
  year      = {2019},
}
```
