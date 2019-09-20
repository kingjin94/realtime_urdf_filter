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
		if np.linalg.norm(np.asarray(jstate.velocity)[2:9]) > 1e-1:
			rospy.logdebug("!!!To fast!!! {} | {}".format(np.linalg.norm(jstate.velocity), np.asarray(jstate.velocity)[2:9]))
			return
			
		cv_image = self.bridge.imgmsg_to_cv2(image, desired_encoding="passthrough").copy()
		cv_image = cv2.resize(cv_image, (240, 180), interpolation=cv2.INTER_NEAREST)  # TODO: Define and use a central point where voxel size, max distance and viewfield are given as a rosparam and adapt along the image chain
		mask = cv_image == self.value_to_replace # get robot pixels 
		mask = binary_dilation(mask, structure=disk(10)) # grow regions of robot pixels
		cv_image[mask] = np.nan # Replace extended robot pixels with nan
		mask =  cv_image < 0.1 # remove close pixels where filter fails
		cv_image[mask] = np.nan

		try:
			new_img = self.bridge.cv2_to_imgmsg(cv_image, image.encoding)
			new_img.header = image.header
			new_info = deepcopy(info)
			new_info.header = info.header
			new_info.height = 180
			new_info.width = 240
			new_info.K = np.asarray(new_info.K)
			new_info.K[0] = new_info.K[0] * 240.0/1024 # from https://answers.opencv.org/question/150551/how-does-resizing-an-image-affect-the-intrinsics/
			new_info.K[4] = new_info.K[4] * 240.0/1024
			new_info.K[2] = new_info.K[2] * 240.0/1024
			new_info.K[5] = new_info.K[5] * 240.0/1024
			new_info.P = np.asarray(new_info.P)
			new_info.P[0] = new_info.P[0] * 240.0/1024 
			new_info.P[2] = new_info.P[2] * 240.0/1024
			new_info.P[5] = new_info.P[5] * 240.0/1024
			new_info.P[6] = new_info.P[6] * 240.0/1024
			
			self.image_pub.publish(new_img)
			rospy.logdebug("Image sent")
			self.info_pub.publish(new_info)
			rospy.logdebug("Camera Info sent")
		except CvBridgeError as e:
			rospy.logerr(e)


"""
Nanifier grabs the result of the URDF filter, which has set the robot pixels in the depth image to "filter_replace_value" on the rosparam server
It ensures that only scans in a moment without movement are used, s.t. the integration can work with a reliable camera pose
It replaces filter_replace_value with NaN, such that these pixels are ignored in the following transformation to a point cloud (node "cloudify")
It also enlarges the robot regions a bit to remove artifacts and resizes the image, s.t. the resolution equals the voxel size at the maximum integration distance
"""
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
