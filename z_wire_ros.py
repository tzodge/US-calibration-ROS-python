from __future__ import division, print_function
import numpy as np
import rospy
from geometry_msgs.msg import TransformStamped, Point
from sensor_msgs.msg import Image
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker
import message_filters
from frame_from_coords_clean import img_3d_coords_zFrame, find_img_frame_axis, find_img_frame_position, centralize_and_scale 
from annotate_image_data_clean import select_points, find_centroids, find_closest_point
from cv_bridge import CvBridge
import transforms3d as t3d
import cv2

_ultrasound_image = None
_robot_pose = None
_new_pose_available = False
_bridge = CvBridge()

def pose_msg_to_transf(msg):
    # print(pose_vect,"pose_vect")
    T = np.eye(4)
    q = [msg.transform.rotation.w, msg.transform.rotation.x, msg.transform.rotation.y, msg.transform.rotation.z]
    R = t3d.quaternions.quat2mat(q)
    t = [msg.transform.translation.x, msg.transform.translation.y, msg.transform.translation.z]
    T[0:3,0:3] = R
    T[0:3,3] = t
    return T

def _sync_cb(robot_pose, ultrasound_image):
	global _ultrasound_image, _robot_pose, _new_pose_available
	_ultrasound_image = _bridge.imgmsg_to_cv2(ultrasound_image)
	_robot_pose = pose_msg_to_transf(robot_pose)
	_new_pose_available = True

def make_pose_marker(mat, scale = 1):
	marker = Marker()
	marker.header.frame_id = "base"
	marker.header.stamp = rospy.Time.now()
	marker.ns = "z_wire"
	marker.id = 0
	marker.action = Marker.MODIFY
	marker.type = Marker.LINE_LIST
	marker.pose.position.x = mat[0,3] * scale
	marker.pose.position.y = mat[1,3] * scale
	marker.pose.position.z = mat[2,3] * scale
	q = t3d.quaternions.mat2quat(mat[0:3,0:3])
	marker.pose.orientation.x = q[1]
	marker.pose.orientation.y = q[2]
	marker.pose.orientation.z = q[3]
	marker.pose.orientation.w = q[0]
	marker.color.a = 1.0
	marker.color.r = 0.0
	marker.color.g = 1.0
	marker.color.b = 0.0
	marker.scale.x = 0.001 * scale
	marker.points = [Point() for p in range(6)]
	marker.points[1].x = 0.005 * scale
	marker.points[3].y = 0.005 * scale
	marker.points[5].z = 0.005 * scale
	marker.colors = [ColorRGBA(1,0,0,1), ColorRGBA(1,0,0,1),
					 ColorRGBA(0,1,0,1), ColorRGBA(0,1,0,1),
					 ColorRGBA(0,0,1,1), ColorRGBA(0,0,1,1)]
	return marker

_wires = np.array([[0.000, 0.000, 0.000],[0.040, 0.000, 0.000],
				   [0.000, 0.005, 0.000],[0.040, 0.015, 0.000],
                   [0.000, 0.020, 0.000],[0.040, 0.020, 0.000],
                   [0.000, 0.000,-0.010],[0.040, 0.000,-0.010],
                   [0.000, 0.015,-0.010],[0.040, 0.005,-0.010],
                   [0.000, 0.020,-0.010],[0.040, 0.020,-0.010],
                   [0.000, 0.005,-0.020],[0.040, 0.005,-0.020],
                   [0.000, 0.010,-0.020],[0.040, 0.015,-0.020],
                   [0.000, 0.020,-0.020],[0.040, 0.020,-0.020]])

def make_wire_marker(mat, scale = 1):
	marker = Marker()
	marker.header.frame_id = "base"
	marker.header.stamp = rospy.Time.now()
	marker.ns = "z_wire_phantom"
	marker.id = 0
	marker.action = Marker.MODIFY
	marker.type = Marker.LINE_LIST
	marker.pose.position.x = mat[0,3] * scale
	marker.pose.position.y = mat[1,3] * scale
	marker.pose.position.z = mat[2,3] * scale
	q = t3d.quaternions.mat2quat(mat[0:3,0:3])
	marker.pose.orientation.x = q[1]
	marker.pose.orientation.y = q[2]
	marker.pose.orientation.z = q[3]
	marker.pose.orientation.w = q[0]
	marker.color.a = 1.0
	marker.color.r = 0.5
	marker.color.g = 0.5
	marker.color.b = 1.0
	marker.scale.x = 0.001 * scale
	wires = _wires * scale
	marker.points = [Point(p[0], p[1], p[2]) for p in wires]
	marker.colors = [marker.color for c in range(len(marker.points))]
	return marker


def main():
	global _ultrasound_image, _robot_pose, _new_pose_available
	rospy.init_node('z_wire_to_pose', anonymous=True)
	marker_pub = rospy.Publisher('z_wire_pose', Marker, queue_size = 1)
	marker_pub_phantom = rospy.Publisher('z_wire_phantom', Marker, queue_size = 1)
	robo_sub = message_filters.Subscriber('/pose_ee', TransformStamped)
	ultrasound_sub = message_filters.Subscriber('/ultrasound/image_raw', Image)
	ts = message_filters.ApproximateTimeSynchronizer([robo_sub, ultrasound_sub], 10, 1/15)
	ts.registerCallback(_sync_cb)

	last_im_points = np.zeros((9,2))
	# last_im_points = [[521, 261],
	# 				  [722, 261],
	# 				  [797, 262],
	# 				  [523, 366],
	# 				  [603, 365],
	# 				  [797, 363],
	# 				  [592, 471],
	# 				  [727, 468],
	# 				  [794, 469]]
	points_tracked = [False] * 9
	last_tracked = 100
	rate = rospy.Rate(30)
	centroids = []


	def on_click(event, x, y, p1, p2):
		n_tracked_points =  np.sum(np.equal(points_tracked, True))
		no_free_centroids = len(centroids) <= n_tracked_points
		all_points_tracked = n_tracked_points == len(points_tracked)
		if event != cv2.EVENT_LBUTTONDBLCLK or no_free_centroids or all_points_tracked:
			return
		print("Clicked %d, %d" % (x,y))
		try:
			idx = points_tracked.index(False)
		except ValueError:
			return
		points_tracked[idx] = True
		last_im_points[idx] = [x,y]

	print("Starting")
	while not rospy.is_shutdown():
		marker_pub_phantom.publish(make_wire_marker(np.eye(4), 10))
		rate.sleep()
		if not _new_pose_available:
			continue

		centroids = find_centroids(_ultrasound_image)

		for pt in centroids:
			cv2.circle(_ultrasound_image, tuple(pt.astype(int)), 3, color=(255,0,0), thickness=1)

		last_points_tracked = [b for b in points_tracked]

		for idx, pt in enumerate(last_im_points):
			if points_tracked[idx]:
				new_pt = find_closest_point(centroids, pt)
				if np.linalg.norm(new_pt - pt) < 10:
					last_im_points[idx] = new_pt
				else:
					points_tracked[idx] = False

		# Detect glitch frames
		if last_tracked < 10 and sum(points_tracked) == 0:
			last_tracked = last_tracked + 1
			points_tracked = [b for b in last_points_tracked]
			_new_pose_available = False
			rospy.loginfo("Possible ultrasound glitch. Skipping frame.")
			continue
		elif sum(points_tracked) > 0:
			last_tracked = 0

		if np.sum(np.equal(points_tracked, True))  == len(points_tracked):
			im_points = centralize_and_scale(np.array(last_im_points), np.array([388,134]))
			coords_3d = img_3d_coords_zFrame(im_points)
			# print(last_im_points, im_points, coords_3d)
			if coords_3d is not None:
				pos = find_img_frame_position(im_points, coords_3d)
				rot = find_img_frame_axis(im_points, coords_3d)
				T = np.eye(4)
				T[0:3,0:3] = rot
				T[0:3, 3] = pos
				marker_pub.publish(make_pose_marker(T, 10))
		colors = [np.random.rand(3,)*255 for x in last_im_points]
		for idx, pt, tracked in zip(range(9), last_im_points, points_tracked):
			if tracked:
				color = (0,255,0)
			else:
				color = (0,0,255)
			cv2.putText(_ultrasound_image, str(idx + 1), (int(pt[0]) + 10, int(pt[1]) + 5), cv2.FONT_HERSHEY_SIMPLEX, .5, color=color) 
			cv2.circle(_ultrasound_image, tuple([int(x) for x in pt]), 5, color=color,thickness=2)


		cv2.imshow('Ultrasound Image',_ultrasound_image)
		cv2.setMouseCallback('Ultrasound Image', on_click)
		k = cv2.waitKey(1) & 0xFF
		if k == 27:
			break

		_new_pose_available = False
		# if last_im_points is None:
		# 	last_im_points = select_points(_ultrasound_image)
		# 	if len(last_im_points) < 9:
		# 		last_im_points = None
		# 	continue

		# centroids = find_centroids(_ultrasound_image)
		# _new_pose_available = False
		# for idx, pt in enumerate(last_im_points):
		# 	new_pt = find_closest_point(centroids, pt)
		# 	if np.linalg.norm(new_pt - pt) < 10:
		# 		last_im_points[idx] = new_pt
		# 	else:
		# 		last_im_points = None
		# 		break

		# if last_im_points is None:
		# 	print("Lost track")
		# 	continue

		# for pt in last_im_points:
		# 	color = np.random.rand(3,)*255
		# 	cv2.circle(_ultrasound_image, (centroid[0],centroid[1]), 5, color=color,thickness=2)

		

if __name__ == '__main__':
	main()