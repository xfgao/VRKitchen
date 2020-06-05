This branch impelements the functionality of using a webpage-based user interface to collect demonstrations for the *preparing dish* tasks in VRKitchen.

**Requirements:**
- Window 10
- python 2.7

**Getting Started:**
1. git clone -b webApiExecutable https://github.com/xfgao/VRKitchen.git
2. pip install -r requirements.txt
3. Download the zip file and unzip it into the /Binaries folder:

[Download] https://drive.google.com/file/d/1ceQDni1gzyzc2OcQ3foaejTsqkmqw71B/view?usp=sharing


**To collect demonstrations:**
1. To start the server that hosts the user interface, type the command in your windows powershell: 
```
python Script/interface.py
```
2. Run the executable that contains the VRKitchen platform: Binaries/WindowsNoEditor/VRInteractPlatform.exe
3. Go to http://127.0.0.2:9000/ in your browser and you are good to go
4. Your demonstrations would be saved in the folder: Script/dataset
