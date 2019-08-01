#!/usr/bin/python
import rospy
from sensor_msgs.msg import Image, CameraInfo, JointState
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
import sys
from scipy.ndimage.morphology import binary_dilation
from skimage.morphology.selem import disk
import message_filters
from copy import deepcopy

class Nanifier:
	def __init__(self):
		self.image_pub = rospy.Publisher("/panda/depth_camera/depth_image/filtered/image",Image, queue_size=10)
		self.info_pub = rospy.Publisher("/panda/depth_camera/depth_image/filtered/camera_info",CameraInfo, queue_size=10)
		#self.info_pub = rospy.Publisher("/panda/totalyDifferentCameraInfo",CameraInfo, queue_size=10)
		self.bridge = CvBridge()
		self.image_sub = message_filters.Subscriber("/panda/depth_camera/depth_image/filtered", Image)
		self.info_sub = message_filters.Subscriber("/panda/depth_camera/depth_image/camera_info", CameraInfo)
		self.jstate_sub = message_filters.Subscriber("/joint_states", JointState)
		self.value_to_replace = rospy.get_param('~filter_replace_value')
		rospy.loginfo("Nanifier replacing {} with nan".format(self.value_to_replace))
		self.ts = message_filters.ApproximateTimeSynchronizer([self.image_sub, self.info_sub, self.jstate_sub], 10, 0.1)
		self.ts.registerCallback(self.callback)
		rospy.loginfo("Nanifier init done")
		
	def callback(self, image, info, jstate):
		# only process if moving slowly
		if np.linalg.norm(jstate.velocity) > 5e-2:
			print("!!!To fast!!! {} | {}".format(np.linalg.norm(jstate.velocity), jstate.velocity))
			return
			
		cv_image = self.bridge.imgmsg_to_cv2(image, desired_encoding="passthrough").copy()
		mask = cv_image == self.value_to_replace # get robot pixels
		mask = binary_dilation(mask, structure=disk(10)) # grow regions of robot pixels
		cv_image[mask] = np.nan # Replace extended robot pixels with nan

		try:
			new_img = self.bridge.cv2_to_imgmsg(cv_image, image.encoding)
			new_img.header = image.header
			new_info = deepcopy(info)
			new_info.header = info.header
			self.image_pub.publish(new_img)
			rospy.logdebug("Image sent")
			self.info_pub.publish(new_info)
			rospy.logdebug("Camera Info sent")
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
