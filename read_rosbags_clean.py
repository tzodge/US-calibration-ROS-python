import rospy
import rosbag
import pprint 
import numpy as np
import h5py
import time
import matplotlib.pyplot as plt
import sys 

import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


def store_posedata(bag,f_pose):
	t_last = 0

	count =0
	t_array_pose = []
	pose_plot =[]
	msg_value = []

	# for topic, msg, t in bag.read_messages(topics=['/tf']):
	for topic, msg, t in bag.read_messages(topics=['/pose_ee']):

		tf_BE = msg.transform 
		# print(msg.transform.translation,"msg.transform.translation")
		t_secs = msg.header.stamp.secs
		t_nsecs = msg.header.stamp.nsecs
		# print(t_secs)
		# if int(t_secs)/1e9-int(t_last)/1e9>10:
		# 	print("Hi")

		msg_value.append(np.array([count, t_secs,t_nsecs,
									tf_BE.translation.x,tf_BE.translation.y,tf_BE.translation.z,
									tf_BE.rotation.x,tf_BE.rotation.y,tf_BE.rotation.z,tf_BE.rotation.w]).reshape(1,-1)) 

		pose_plot.append([t_secs, tf_BE.translation.x,tf_BE.translation.y,tf_BE.translation.z,\
									tf_BE.rotation.x,tf_BE.rotation.y,tf_BE.rotation.z,tf_BE.rotation.w])
		print("pose_count: ",  count)

		count+=1

		t_last = t_secs
		t_array_pose.append(t_secs)




	msg_value = np.concatenate(msg_value)
	dset = f_pose.create_dataset("pose_data", data=msg_value)


	pose_plot_array = np.array(pose_plot)
	print(pose_plot_array.shape,"pose_plot_array.shape")
	# plt.plot(range(len(pose_plot_array)), pose_plot_array[:,1], label='1')
	# plt.plot(range(len(pose_plot_array)), pose_plot_array[:,2], label='2')
	# plt.plot(range(len(pose_plot_array)), pose_plot_array[:,3], label='3')
	# plt.plot(range(len(pose_plot_array)), pose_plot_array[:,4], label='4')
	# plt.plot(range(len(pose_plot_array)), pose_plot_array[:,5], label='5')
	# plt.plot(range(len(pose_plot_array)), pose_plot_array[:,6], label='6')
	# plt.show()
	return t_array_pose


def store_imagestamps(bag,f_image):
	t_last = 0

	count=0
	msg_value = []
	t_array_image = []
	## reading the image messages and saving their index in an h5 file
	for topic, msg, t in bag.read_messages(topics=['/camera/image_raw']):
 		t_secs = msg.header.stamp.secs
		t_nsecs = msg.header.stamp.nsecs
		# print(t_secs)
		if int(t_secs)/1e9-int(t_last)/1e9>10:
			print("Hi")
 
		msg_value.append(np.array([count,t_secs,t_nsecs]).reshape(1,-1))
 		count+=1
		t_array_image.append(t_secs)
		print("image_count: ",  count)
 
		t_last = t_secs
 
	msg_value = np.concatenate(msg_value)
	# print(msg_value.shape)
	dset = f_image.create_dataset("image_data", data=msg_value)


	# bag.close()

	# plt.plot(t_array_pose,range(len(t_array_pose)),label='pose_data')
	# plt.plot(t_array_image,range(len(t_array_image)),label='image_data')
	# plt.legend()
	# plt.show()

	return t_array_image

def store_images(bag,images_folder):
    bridge = CvBridge()
    count = 1
    for topic, msg, t in bag.read_messages(topics=["/camera/image_raw"]):
        cv2_img = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        cv2.imwrite(images_folder+'/image_{}.jpg'.format(count), cv2_img)

        # cv2.imwrite(os.path.join(args.output_dir, "frame%06i.png" % count), cv_img)
        print "Wrote image %i" % count

        count += 1



if __name__ == '__main__':
	file_identifier = '2020-09-17-18-34-53'
	prefix = './rosbags/calibration_bag_scaling_'

	filename = prefix+file_identifier+'.bag'
	bag = rosbag.Bag(filename)

	
	f_pose = h5py.File("./h5_files/pose_"+ file_identifier +".h5", "w")
	f_image = h5py.File("./h5_files/image_"+ file_identifier +".h5", "w")

	# store_posedata(bag,f_pose)
	store_imagestamps(bag,f_image)
	# store_images(bag,"./temp_images")