#!/usr/bin/env python  
import rospy
import math
import tf2_ros
import tf2_msgs.msg
import geometry_msgs.msg

baxter_mesh_list = ['head', 'torso', 'screen', 'left_arm_mount',\
'left_upper_shoulder', 'left_lower_shoulder', 'left_upper_elbow',\
'left_lower_elbow', 'left_upper_forearm', 'left_lower_forearm',\
'left_wrist', 'right_arm_mount','right_upper_shoulder', \
'right_lower_shoulder', 'right_upper_elbow','right_lower_elbow',\
 'right_upper_forearm', 'right_lower_forearm', 'right_wrist',]

if __name__ == '__main__':
	rospy.init_node('tf2_listener')
	tfBuffer = tf2_ros.Buffer()
	listener = tf2_ros.TransformListener(tfBuffer)
	pub_tf = rospy.Publisher("/tf", tf2_msgs.msg.TFMessage, queue_size=1)
	rate = rospy.Rate(100)
	while not rospy.is_shutdown():
		try:
			t = rospy.Time()
			for mesh in baxter_mesh_list:
				trans = tfBuffer.lookup_transform('base', mesh, t, rospy.Duration(1.0))
				trans.header.frame_id = "ue_base"
				trans.child_frame_id = "ue_"+mesh
				trans.transform.translation.x *= 100
				trans.transform.translation.y *= 100
				trans.transform.translation.z *= 100
				tfm = tf2_msgs.msg.TFMessage([trans])
				pub_tf.publish(tfm)
			rate.sleep()
		except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
			print("Fail to broadcast transform")
			rate.sleep()
			continue


	rospy.spin()
		