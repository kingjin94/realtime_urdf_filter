#!/usr/bin/python
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
import sys

class Nanifier:
	def __init__(self):
		self.image_pub = rospy.Publisher("/panda/depth_camera/depth_image/filtered_nan",Image, queue_size=10)
		self.bridge = CvBridge()
		self.image_sub = rospy.Subscriber("/panda/depth_camera/depth_image/filtered", Image, self.callback)
		self.value_to_replace = rospy.get_param('~filter_replace_value')
		rospy.loginfo("Nanifier replacing {} with nan".format(self.value_to_replace))
		rospy.loginfo("Nanifier init done")
		
	def callback(self, msg):
		cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough").copy()
		cv_image[cv_image == self.value_to_replace] = np.nan
		try:
			new_msg = self.bridge.cv2_to_imgmsg(cv_image, msg.encoding)
			new_msg.header = msg.header
			self.image_pub.publish(new_msg)
			rospy.logdebug("Image sent")
		except CvBridgeError as e:
			rospy.logerr(e)


def main(args):
	rospy.init_node('nanify')
	nanifier = Nanifier()

	try:
		rospy.spin()
	except KeyboardInterrupt:
		rospy.loginfo("Nanifier shutting down")
	cv2.destroyAllWindows()

if __name__ == '__main__':
	main(sys.argv)
