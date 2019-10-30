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
from mask_rcnn_ros.msg import Result
import matplotlib.pyplot as plt

class Nanifier:
	def __init__(self, depth_img_topic, depth_img_info, depth_scale):
		self.depth_scale = float(depth_scale)
		self.image_pub = rospy.Publisher("/panda/depth_camera/depth_image/filtered/image",Image, queue_size=10)
		self.rgb_image_pub = rospy.Publisher("/panda/depth_camera/image/filtered/seg",Image, queue_size=10)
		self.info_pub = rospy.Publisher("/panda/depth_camera/depth_image/filtered/camera_info",CameraInfo, queue_size=10)
		#self.info_pub = rospy.Publisher("/panda/totalyDifferentCameraInfo",CameraInfo, queue_size=10)
		self.bridge = CvBridge()
		self.image_sub = message_filters.Subscriber(depth_img_topic, Image) # old: "/panda/depth_camera/depth_image/filtered"
		self.seg_res_sub = message_filters.Subscriber("/panda/depth_camera/image/seg_res", Result)
		self.info_sub = message_filters.Subscriber(depth_img_info, CameraInfo) # old: "/panda/depth_camera/depth_image/camera_info"
		self.jstate_sub = message_filters.Subscriber("/joint_states", JointState)
		self.value_to_replace = 50.0 #rospy.get_param('~filter_replace_value')
		rospy.loginfo("Nanifier replacing {} with nan".format(self.value_to_replace))
		self.ts = message_filters.ApproximateTimeSynchronizer([self.image_sub, self.info_sub, self.jstate_sub, self.seg_res_sub], 10, 1.0)
		self.ts.registerCallback(self.callback)
		rospy.loginfo("Nanifier init done")
		
	def filter_seg_res(self, res_msg, classes, replaceValue):
		"""
		Finds all instances of the the classes given within res_msg, overlays their masks and returns an image containing all instances marked with replaceValue
		
		* res_msg: mask_rcnn_ros.msg.Result
		* classes: list of strings
		* replaceValue: greyscale / color code as numpy array; rest will default to black / 0
		"""
		indices_of_interest = [idx for idx, elem in enumerate(res_msg.class_names) if elem in classes]
		if not indices_of_interest:
			#print("No object of interest")
			if len(res_msg.masks) > 0:
				img_res = np.asarray((res_msg.masks[0].height, res_msg.masks[0].width))
				return np.zeros(np.concatenate((img_res, replaceValue.shape)), dtype=replaceValue.dtype)
			else:
				return np.zeros((786, 1024, 3), dtype=replaceValue.dtype)
		#print("Found {} objects of interest".format(len(indices_of_interest)))
		res = np.zeros((res_msg.masks[indices_of_interest[0]].height, res_msg.masks[indices_of_interest[0]].width), dtype=bool)
		
		for idx in indices_of_interest:
			seg_mask = self.bridge.imgmsg_to_cv2(res_msg.masks[idx], desired_encoding="passthrough").copy()
			res=np.logical_or(res, seg_mask)
			
		ret_img = np.zeros(np.concatenate((res.shape, replaceValue.shape)), dtype=replaceValue.dtype)
		ret_img[res,:] = replaceValue
		return ret_img
		
	def callback(self, image, info, jstate, seg_res_msg):
		# only process if moving slowly
		if np.linalg.norm(np.asarray(jstate.velocity)[2:9]) > 1e-1:
			rospy.logdebug("!!!To fast!!! {} | {}".format(np.linalg.norm(jstate.velocity), np.asarray(jstate.velocity)[2:9]))
			return
			
		cv_image = self.depth_scale*self.bridge.imgmsg_to_cv2(image, desired_encoding="32FC1").copy()
		old_shape = cv_image.shape
		cv_image = cv2.resize(cv_image, (240, 180), interpolation=cv2.INTER_NEAREST)  # TODO: Define and use a central point where voxel size, max distance and viewfield are given as a rosparam and adapt along the image chain
		mask = np.bitwise_or(np.bitwise_and(self.value_to_replace-0.1 < cv_image, cv_image < self.value_to_replace+0.1), (cv_image < 0.1)) # get robot and invalid pixels 
		mask = binary_dilation(mask, structure=disk(10)) # grow regions of robot pixels
		# mask = binary_dilation(mask, structure=disk(25)) # grow regions of robot pixels
		cv_image[mask] = np.nan # Replace extended robot pixels with nan
		# mask =  cv_image < 0.1 # remove close pixels where filter fails
		# cv_image[mask] = np.nan
		
		# cv_rgb_image = self.bridge.imgmsg_to_cv2(rgb_image, desired_encoding="passthrough").copy()
		# cv_rgb_image = cv2.resize(cv_rgb_image, (240, 180), interpolation=cv2.INTER_NEAREST)
		cv_seg_image_cans = self.filter_seg_res(seg_res_msg, ["bottle", "cup"], np.asarray([255, 0, 0], dtype=np.uint8))
		cv_seg_image_table = self.filter_seg_res(seg_res_msg, ["dining table"], np.asarray([0, 255, 0], dtype=np.uint8))
		cv_seg_image = cv_seg_image_cans
		cv_seg_image[cv_seg_image_cans[:,:,0]==0] = cv_seg_image_table[cv_seg_image_cans[:,:,0]==0] # Table is in the background # Be more clever --> bigger objects must be farther in the background?!
		cv_seg_image = cv2.resize(cv_seg_image, (240, 180), interpolation=cv2.INTER_NEAREST)
		
		try:
			new_depth_img_msg = self.bridge.cv2_to_imgmsg(cv_image, "32FC1")
			new_depth_img_msg.header = image.header
			new_depth_img_msg.header.frame_id = "panda/panda_camera"
			new_seg_image_msg = self.bridge.cv2_to_imgmsg(cv_seg_image, "rgb8")
			new_seg_image_msg.header = seg_res_msg.header
			new_seg_image_msg.header.frame_id = "panda/panda_camera"
			new_info = deepcopy(info)
			new_info.header = info.header
			new_info.header.frame_id = "panda/panda_camera"
			new_info.height = 180
			new_info.width = 240
			# print("Old image size: {}, new: {}".format(old_shape, cv_image.shape))
			new_info.K = np.asarray(new_info.K)
			new_info.K[0] = new_info.K[0] * 240.0/old_shape[1] # from https://answers.opencv.org/question/150551/how-does-resizing-an-image-affect-the-intrinsics/
			new_info.K[4] = new_info.K[4] * 180.0/old_shape[0]
			new_info.K[2] = new_info.K[2] * 240.0/old_shape[1]
			new_info.K[5] = new_info.K[5] * 180.0/old_shape[0]
			new_info.P = np.asarray(new_info.P)
			new_info.P[0] = new_info.P[0] * 240.0/old_shape[1] 
			new_info.P[2] = new_info.P[2] * 240.0/old_shape[1]
			new_info.P[5] = new_info.P[5] * 180.0/old_shape[0]
			new_info.P[6] = new_info.P[6] * 180.0/old_shape[0]
			
			self.image_pub.publish(new_depth_img_msg)
			rospy.logdebug("Image sent")
			self.rgb_image_pub.publish(new_seg_image_msg)
			rospy.logdebug("RGB Image sent")
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
	myargv = rospy.myargv(argv=sys.argv)
	if len(myargv) != 4:
		print("Usage: nanify depth_img depth_img_info depth_scale_to_m")
		return -1
	nanifier = Nanifier(myargv[1], myargv[2], myargv[3])

	try:
		rospy.spin()
	except KeyboardInterrupt:
		rospy.loginfo("Nanifier shutting down")
	cv2.destroyAllWindows()

if __name__ == '__main__':
	main(sys.argv)
