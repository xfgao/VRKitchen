This branch impelements the functionality of using a webpage-based user interface to collect demonstrations for the *preparing dish* tasks in VRKitchen.

**Requirements:**
- Window 10
- Miniconda2 (https://docs.conda.io/en/latest/miniconda.html)
	- **Make sure that conda is added to your PATH environment variable**

**Getting Started:**
1. git clone -b webApiExecutable https://github.com/xfgao/VRKitchen.git
2. In your **anaconda prompt**, type:
```
conda env create -f environment.yml  
conda activate kitchen_env
pip install pyrapidjson
```
3. Download the zip file and unzip it into the /Binaries folder:
[Download] https://drive.google.com/file/d/1kwn-DkoPHyQC2iTcXkizLGSi7chUbdYs/view?usp=sharing

**To collect demonstrations:**
1. To start the server that hosts the user interface, type the command in your anaconda prompt. When you see the prompt "Server starts to listen", go to step 2.
```
conda activate kitchen_env
python Script/interface.py
```
2. Run the executable that contains the VRKitchen platform, press \` to release your mouse from the game, and go to step 3 immediately after you release your mouse.
```
Binaries/WindowsNoEditor/VRInteractPlatform.exe
```
3. Go to http://127.0.0.2:9000/ in your browser, you will use the webpage to control your agent.
4. Your demonstrations would be saved in the folder: Script/dataset. You can zip and send the whole /dataset folder back to us.
