#!/usr/bin/env python
from rapidjson import loads,dumps 
from socketClient import Client
from socketServer import Server
import rospy
import math
import tf2_ros
import tf2_msgs.msg
import geometry_msgs.msg
import numpy as np
from PIL import Image
from io import BytesIO

baxter_mesh_list = ['base', 'head', 'torso', 'left_arm_mount',\
'left_upper_shoulder', 'left_lower_shoulder', 'left_upper_elbow',\
'left_lower_elbow', 'left_upper_forearm', 'left_lower_forearm',\
'left_wrist', 'right_arm_mount','right_upper_shoulder', \
'right_lower_shoulder', 'right_upper_elbow','right_lower_elbow',\
 'right_upper_forearm', 'right_lower_forearm', 'right_wrist']

if __name__ == '__main__':
	rospy.init_node('tf2_listener')
	tfBuffer = tf2_ros.Buffer()
	listener = tf2_ros.TransformListener(tfBuffer)
	rate = rospy.Rate(30)
	endpoint = ("128.97.86.170", 10020)
	client = Client(endpoint)
	serverpoint = ("128.97.86.192", 10021)
	server = Server(serverpoint)
	server.listen()
	frame = 1
	
	while not rospy.is_shutdown():
		if not client.connect():
			continue
		data = {}
		data['RobotName'] = 'BaxterRobot'
		data['Anim'] = []
		try:
			i = 0
			for mesh in baxter_mesh_list:	
				t = tfBuffer.lookup_transform('world', 'ue_'+mesh, rospy.Time())
				data['Anim'].append({})
				data['Anim'][i]['MeshPose'] = \
				{'Loc':{"X":t.transform.translation.x, "Y":t.transform.translation.y, "Z":t.transform.translation.z},\
				 'Rot':{"X":t.transform.rotation.x, "Y":t.transform.rotation.y, "Z":t.transform.rotation.z, "W":t.transform.rotation.w}}
				data['Anim'][i]['MeshName'] = mesh
				i += 1

			# print(data)
			msg = dumps(data)
			msg = msg+"\n"
			client.send(msg.encode())
			# wait for the data
			# rate.sleep()	

			depth_head = server.getBuffer()
			print(depth_head)
			assert(depth_head == "Depth")
			depth_data = np.load(BytesIO(server.getBuffer()))
			depth_image = Image.fromarray(depth_data)
			# depth_image.show()
			# depth_image.close()
			
			rgb_head = server.getBuffer()
			print(rgb_head)
			assert(rgb_head == "RGB")
			rgb_data = np.load(BytesIO(server.getBuffer()))
			rgb_image = Image.fromarray(rgb_data)
			# rgb_image.show()
			# rgb_image.close()

			done_head = server.getBuffer()
			print(done_head)
			assert(done_head == "Done")
			done = (server.getBuffer() == "1")
			# print(type(done))
			print(done)

			print(frame)
			frame += 1
			# if done == True:
				# break
			

		except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
			# print("something wrong")
			rate.sleep()
			continue
		# i += 1


	# rospy.spin()
