#!/usr/bin/env python  
import rospy
import tf2_ros
import tf
import tf2_msgs.msg
import geometry_msgs.msg
import math
import numpy as np
import copy
import time

baxter_mesh_list = ['head', 'torso', 'screen', 'left_arm_mount',\
'left_upper_shoulder', 'left_lower_shoulder', 'left_upper_elbow',\
'left_lower_elbow', 'left_upper_forearm', 'left_lower_forearm',\
'left_wrist', 'right_arm_mount','right_upper_shoulder', \
'right_lower_shoulder', 'right_upper_elbow','right_lower_elbow',\
 'right_upper_forearm', 'right_lower_forearm', 'right_wrist',]


class baxter_ue_base_publisher(object):
	def __init__(self, pose):
		self.pose = copy.copy(pose)
		self.pub_tf = rospy.Publisher("/tf", tf2_msgs.msg.TFMessage, queue_size=1)
		self.rate = 200
		self.tfBuffer = tf2_ros.Buffer()
		self.listener = tf2_ros.TransformListener(self.tfBuffer)

	def pub_ue_base_pose(self, event):
		t = geometry_msgs.msg.TransformStamped()
		t.header.stamp = rospy.Time.now()
		t.header.frame_id = "world"
		t.child_frame_id = "ue_base"
		# print(self.pose)
		t.transform.translation.x = self.pose['Loc']['X']
		t.transform.translation.y = self.pose['Loc']['Y']
		t.transform.translation.z = self.pose['Loc']['Z']
		t.transform.rotation.x = self.pose['Rot']['X']
		t.transform.rotation.y = self.pose['Rot']['Y']
		t.transform.rotation.z = self.pose['Rot']['Z']
		t.transform.rotation.w = self.pose['Rot']['W']
		tfm = tf2_msgs.msg.TFMessage([t])
		self.pub_tf.publish(tfm)

		for mesh in baxter_mesh_list:
			trans = self.tfBuffer.lookup_transform('base', mesh, rospy.Time(0), rospy.Duration(1.0))
			trans.header.frame_id = "ue_base"
			trans.child_frame_id = "ue_"+mesh
			trans.transform.translation.x *= 100
			trans.transform.translation.y *= -100
			trans.transform.translation.z *= 100
			trans.transform.rotation.y *= -1
			trans.transform.rotation.w *= -1
			tfm = tf2_msgs.msg.TFMessage([trans])
			self.pub_tf.publish(tfm)
		# q = tf.transformations.quaternion_from_euler(msg['Rot']['X'], msg['Rot']['Y'], msg['Rot']['Z'])
		# t.transform.rotation.x = q[0]
		# t.transform.rotation.y = q[1]
		# t.transform.rotation.z = q[2]
		# t.transform.rotation.w = q[3]

	def change_loc(self, new_loc):
		self.pose['Loc'] = new_loc

	def change_rot(self, new_rot):
		self.pose['Rot'] = new_rot

	def start_pub(self):
		self.timer = rospy.Timer(rospy.Duration(1.0/self.rate), self.pub_ue_base_pose)

	def print_pose(self):
		print(self.pose)

# first rotate, go next
def baxter_straight_walk(start, end, lin_vel, rot_vel, base_pub=None):
	if base_pub == None:
		base_pub = baxter_ue_base_publisher(start)
	rate = 50
	r = rospy.Rate(rate)

	x1 = start['Loc']['X']
	y1 = start['Loc']['Y']
	z1 = start['Loc']['Z']
	x2 = end['Loc']['X']
	y2 = end['Loc']['Y']
	z2 = end['Loc']['Z']

	if not z1 == z2:
		print("baxter cannot jump!")
		return False

	# first rotate to end direction
	theta_end = end['Theta']

	theta_start = start['Theta']

	print(theta_start)
	print(theta_end)
	base_pub.start_pub()

	if theta_end < theta_start:
		r_vel = -rot_vel/rate
	else:
		r_vel = rot_vel/rate

	for angle in np.arange(theta_start, theta_end+r_vel, r_vel):
		q = tf.transformations.quaternion_from_euler(0, 0, angle)
		new_rot = {'X':q[0], 'Y':q[1], 'Z':q[2], 'W':q[3]}
		base_pub.change_rot(new_rot)
		# base_pub.print_pose()
		r.sleep()

	# then go ahead
	if x2 < x1:
		x_vel = -lin_vel/rate
	else:
		x_vel = lin_vel/rate

	if y2 < y1:
		y_vel = -lin_vel/rate
	else:
		y_vel = lin_vel/rate

	num = np.max([(x2-x1)//x_vel, (y2-y1)//y_vel])
	x_locs = np.linspace(x1, x2, num)
	y_locs = np.linspace(y1, y2, num)
	for i in range(len(x_locs)):
		new_loc = {'X':x_locs[i], 'Y':y_locs[i], 'Z':z1}
		base_pub.change_loc(new_loc)
		r.sleep()

	return base_pub


# if __name__ == '__main__':
	# rospy.init_node('ue_broadcaster')
	# # go to cupboard
	# # start1 = {'Loc':{"X":-910, "Y":-50, "Z":130}, 'Rot':{"X":0, "Y":0, "Z":0.707, "W":0.707}}
	# # end1 = {'Loc':{"X":-910, "Y":317, "Z":130}, 'Rot':{"X":0, "Y":0, "Z":0.707, "W":0.707}}
	# # lin_vel = 10
	# # rot_vel = 0.1
	# # baxter_straight_walk(start, end, lin_vel, rot_vel)
	# start2 = {'Loc':{"X":-910, "Y":317, "Z":130}, 'Rot':{"X":0, "Y":0, "Z":0.707, "W":0.707}}
	# base_pub = baxter_ue_base_publisher(start2)
	# base_pub.start_pub()
	# rospy.spin()