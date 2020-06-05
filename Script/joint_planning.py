import argparse
import sys
import math
from ue_base_bc import baxter_ue_base_publisher
from ue_base_bc import baxter_straight_walk
from copy import copy
import rospy
import actionlib
from control_msgs.msg import (
	FollowJointTrajectoryAction,
	FollowJointTrajectoryGoal,
)
from trajectory_msgs.msg import (
	JointTrajectoryPoint,
)
import baxter_interface
from baxter_interface import CHECK_VERSION

class Trajectory(object):
	def __init__(self, limb):
		ns = 'robot/limb/' + limb + '/'
		self._client = actionlib.SimpleActionClient(
			ns + "follow_joint_trajectory",
			FollowJointTrajectoryAction,
		)
		self._goal = FollowJointTrajectoryGoal()
		self._goal_time_tolerance = rospy.Time(5)
		self._goal.goal_time_tolerance = self._goal_time_tolerance
		server_up = self._client.wait_for_server(timeout=rospy.Duration(10.0))
		if not server_up:
			rospy.logerr("Timed out waiting for Joint Trajectory"
						 " Action Server to connect. Start the action server"
						 " before running example.")
			rospy.signal_shutdown("Timed out waiting for Action Server")
			sys.exit(1)
		self.clear(limb)

	def add_point(self, positions, time):
		point = JointTrajectoryPoint()
		point.positions = copy(positions)
		point.time_from_start = rospy.Duration(time)
		self._goal.trajectory.points.append(point)

	def start(self):
		self._goal.trajectory.header.stamp = rospy.Time.now()
		self._client.send_goal(self._goal)

	def stop(self):
		self._client.cancel_goal()

	def wait(self, timeout=15.0):
		self._client.wait_for_result(timeout=rospy.Duration(timeout))

	def result(self):
		return self._client.get_result()

	def clear(self, limb):
		self._goal = FollowJointTrajectoryGoal()
		self._goal.goal_time_tolerance = self._goal_time_tolerance
		self._goal.trajectory.joint_names = [limb + '_' + joint for joint in \
			['s0', 's1', 'e0', 'e1', 'w0', 'w1', 'w2']]


def main():
	"""RSDK Joint Trajectory Example: Simple Action Client
	Creates a client of the Joint Trajectory Action Server
	to send commands of standard action type,
	control_msgs/FollowJointTrajectoryAction.
	Make sure to start the joint_trajectory_action_server.py
	first. Then run this example on a specified limb to
	command a short series of trajectory points for the arm
	to follow.
	"""
	
	rospy.init_node("rsdk_joint_trajectory_client")
	rs = baxter_interface.RobotEnable(CHECK_VERSION)
	rs.enable()
	limb = "right"
	traj = Trajectory(limb)
	rospy.on_shutdown(traj.stop)
	limb_interface = baxter_interface.limb.Limb(limb)
	limb_left = "left"
	traj_left = Trajectory(limb_left)
	rospy.on_shutdown(traj_left.stop)
	limb_interface_left = baxter_interface.limb.Limb(limb_left)

	lin_vel = 50
	rot_vel = 0.5

	# arms to neutral
	ptuck_right = [-0.6319326229932258, -1.2891348487210044, \
	2.169440285035634, 2.563619022432359,-0.41563579747459034, \
	0.7554430205308149, 0.7919724401314436]

	ptuck_left =  [0.7213134137014734, -1.247332705896941, \
	-2.217926584442605, 2.5998647432754964, 0.4485462863317595, \
	0.7624866086179427, -0.7808816527203533]

	traj.add_point(ptuck_right, 5)
	traj.start()
	traj.wait(5.0)
	traj.clear(limb)
	traj_left.add_point(ptuck_left, 5)
	traj_left.start()
	traj_left.wait(5.0)
	traj_left.clear(limb_left)

	# test
	# rospy.sleep(20.0)
	start = {'Loc':{"X":-1010, "Y":-40, "Z":130}, 'Theta':-1/2.0*(math.pi)}
	end = {'Loc':{"X":-1010, "Y":-40, "Z":130}, 'Theta':-1/2.0*(math.pi)}
	# baxter_straight_walk(start, end, lin_vel, rot_vel, pub)
	pub = baxter_straight_walk(start, end, lin_vel, rot_vel)
	p_plate = [ 0.10038920273745244, 1.046894869685846, \
	-1.6469883899815603, 1.9627006211373796, -2.4770295434221783, \
	0.923633806153064, -2.610088362572311]
	traj_left.add_point(p_plate, 10.0)
	traj_left.start()
	traj_left.wait(10.0)
	traj_left.clear(limb_left)	
	# end of test


	# # # wait for human to give order and go to the plate
	# start = {'Loc':{"X":-986, "Y":0, "Z":130},'Theta':0}
	# end = {'Loc':{"X":-986, "Y":0, "Z":130},'Theta':0}
	# pub = baxter_straight_walk(start, end, lin_vel, rot_vel)
	# rospy.sleep(5)

	# start = {'Loc':{"X":-986, "Y":0, "Z":130},'Theta':0}
	# end = {'Loc':{"X":-986, "Y":310, "Z":130}, 'Theta':1/2.0*(math.pi)}
	# baxter_straight_walk(start, end, lin_vel, rot_vel, pub)
	# rospy.sleep(3.)

	# # go to the cabinet
	# start = {'Loc':{"X":-986, "Y":310, "Z":130}, 'Theta':1/2.0*(math.pi)}
	# end = {'Loc':{"X":-910, "Y":310, "Z":130}, 'Theta':0}
	# baxter_straight_walk(start, end, lin_vel, rot_vel, pub)

	# start = {'Loc':{"X":-910, "Y":310, "Z":130},'Theta':0}
	# end = {'Loc':{"X":-910, "Y":310, "Z":130}, 'Theta':1/2.0*(math.pi)}
	# baxter_straight_walk(start, end, lin_vel, rot_vel, pub)

	# # open the door
	# pin = [-0.13033562032701695, -1.2232455882443016, \
	# 	1.6191079054163913, 0.9843120231941045, -0.45944945223521483, \
	# 	2.093954080150631, 0.16988330151054498]
	# traj.add_point(pin, 1)
	# traj.start()
	# traj.wait(2.0)
	# traj.clear(limb)

	# pout = [-0.973426893216705, -1.2188422490394712, \
	# 	2.361740074627363, 1.2632099137204928, -0.3322520643558802, \
	# 	2.074963707680137, 0.7682677240753923]
	# traj.add_point(pout, 1)
	# traj.start()
	# traj.wait(2.0)
	# traj.clear(limb)

	# # pick up the carrot
	# p = [ 0.28745031614438155, -0.3533269215849, \
	# 1.75223601003783, 0.6153310834697461, -1.5305183854398772, \
	# 1.9108530015891674, 0.4618292365996348]
	# traj.add_point(p, 1)
	# traj.start()
	# traj.wait(2.0)
	# traj.clear(limb)

	# traj.add_point(pout, 1)
	# traj.start()
	# traj.wait(2.0)
	# traj.clear(limb)

	# # close the door
	# start = {'Loc':{"X":-910, "Y":310, "Z":130}, 'Theta':1/2.0*(math.pi)}
	# end = {'Loc':{"X":-910, "Y":265, "Z":130}, 'Theta':1/2.0*(math.pi)}
	# baxter_straight_walk(start, end, lin_vel, rot_vel, pub)

	# start = {'Loc':{"X":-910, "Y":265, "Z":130}, 'Theta':1/2.0*(math.pi)}
	# end = {'Loc':{"X":-910, "Y":265, "Z":130}, 'Theta':0}
	# baxter_straight_walk(start, end, lin_vel, rot_vel, pub)

	# start = {'Loc':{"X":-910, "Y":265, "Z":130}, 'Theta':0}
	# end = {'Loc':{"X":-880, "Y":290, "Z":130}, 'Theta':0}
	# baxter_straight_walk(start, end, lin_vel, rot_vel, pub)

	# start = {'Loc':{"X":-880, "Y":290, "Z":130}, 'Theta':0}
	# end = {'Loc':{"X":-920, "Y":290, "Z":130}, 'Theta':1/2.0*(math.pi)}
	# baxter_straight_walk(start, end, lin_vel, rot_vel, pub)


	# # go back and give out the carrot
	# start = {'Loc':{"X":-920, "Y":290, "Z":130}, 'Theta':1/2.0*(math.pi)}
	# end = {'Loc':{"X":-920, "Y":50, "Z":130}, 'Theta':-1/2.0*(math.pi)}
	# baxter_straight_walk(start, end, lin_vel, rot_vel, pub)

	# start = {'Loc':{"X":-920, "Y":50, "Z":130}, 'Theta':-1/2.0*(math.pi)}
	# end = {'Loc':{"X":-920, "Y":50, "Z":130}, 'Theta':0}
	# baxter_straight_walk(start, end, lin_vel, rot_vel, pub)

	# pin = [-0.13033562032701695, -1.2232455882443016, \
	# 	1.6191079054163913, 0.9843120231941045, -0.45944945223521483, \
	# 	2.093954080150631, 0.16988330151054498]
	# traj.add_point(pin, 1)
	# traj.start()
	# traj.wait(2.0)
	# traj.clear(limb)
	# rospy.sleep(5.0)


	# # go to the plate, get it and give it to human
	# # rospy.sleep(20.0)
	# start = {'Loc':{"X":-920, "Y":50, "Z":130}, 'Theta':0}
	# end = {'Loc':{"X":-986, "Y":290, "Z":130}, 'Theta':1/2.0*(math.pi)}
	# baxter_straight_walk(start, end, lin_vel, rot_vel, pub)
	# # pub = baxter_straight_walk(start, end, lin_vel, rot_vel)

	# p_plate = [ 0.10038920273745244, 1.046894869685846, \
	# -1.6469883899815603, 1.9627006211373796, -2.4770295434221783, \
	# 0.923633806153064, -2.610088362572311]
	# traj_left.add_point(p_plate, 10.0)
	# traj_left.start()
	# traj_left.wait(10.0)
	# traj_left.clear(limb_left)	

	# start = {'Loc':{"X":-986, "Y":290, "Z":130}, 'Theta':1/2.0*(math.pi)}
	# end = {'Loc':{"X":-986, "Y":315, "Z":130}, 'Theta':1/2.0*(math.pi)}
	# baxter_straight_walk(start, end, lin_vel, rot_vel, pub)

	# start = {'Loc':{"X":-986, "Y":315, "Z":130}, 'Theta':1/2.0*(math.pi)}
	# end = {'Loc':{"X":-986, "Y":200, "Z":130}, 'Theta':1/2.0*(math.pi)}
	# baxter_straight_walk(start, end, lin_vel, rot_vel, pub)
	# start = {'Loc':{"X":-986, "Y":200, "Z":130}, 'Theta':1/2.0*(math.pi)}
	# end = {'Loc':{"X":-930, "Y":200, "Z":130}, 'Theta':0}
	# baxter_straight_walk(start, end, lin_vel, rot_vel, pub)




	rospy.spin()



if __name__ == "__main__":
	main()