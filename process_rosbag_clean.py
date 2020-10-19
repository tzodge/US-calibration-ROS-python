import sys
import os
import rosbag
import numpy as np
import h5py

from read_rosbags_clean import store_posedata 
from read_rosbags_clean import store_imagestamps 
from read_rosbags_clean import store_images 

from sync_msg_clean import SearchSync

from available_images_clean import available_images

def create_directories(save_dir):
	if not os.path.isdir(save_dir): 
		os.makedirs(save_dir) #start fresh

		image_dir = save_dir+"/calib_images"
		os.mkdir(image_dir)	

		h5_dir = save_dir+"/h5_files"
		os.mkdir(h5_dir)	

		image_frame_coords_dir = save_dir+"/image_frame_coords" 
		os.mkdir(image_frame_coords_dir)	

		extra_files_dir = save_dir+"/extra_files" 
		os.mkdir(extra_files_dir)	

		a = np.array([])
		txt_file_selected_images = extra_files_dir + "/selected_image_ids.txt"
		print (txt_file_selected_images,"txt_file_selected_images")
		np.savetxt(txt_file_selected_images,a)
	else:
		print ("{} already exists".format(save_dir))

def synchronize(save_dir):
	h5_file_pose = save_dir + "/h5_files/pose_data.h5" 
	h5_file_images = save_dir + "/h5_files/image_data.h5" 

	h5_file_pose_sync = save_dir + "/h5_files/pose_data_sync.h5"
	h5_file_images_sync = save_dir + "/h5_files/image_data_sync.h5"
	
	f_pose = h5_file_pose
	f_image = h5_file_images

	f_pose_sync = h5_file_pose_sync
	f_image_sync = h5_file_images_sync

	ss = SearchSync(f_pose,f_image)
	ss.read_msgs()	#micron_coordinates,image_pixel_coordinates
	ss.search_and_sync()
	ss.save_h5(f_pose_sync,f_image_sync)


def process_rosbag(filename, save_dir):
	bag = rosbag.Bag(filename)

	h5_file_pose = h5py.File(save_dir + "/h5_files/pose_data.h5", "w")
	store_posedata(bag,h5_file_pose)
	h5_file_pose.close()


	h5_file_images = h5py.File(save_dir + "/h5_files/image_data.h5", "w")
	store_imagestamps(bag,h5_file_images)
	h5_file_images.close()

	image_dir = save_dir+"/calib_images"
	store_images(bag,image_dir)

	synchronize(save_dir)

	available_images(save_dir)

if __name__ == '__main__':
	args = sys.argv
	if len(args) != 3:
		print( "\n Usage: python process_rosbag.py <rosbag_path> <rosbag_save_path>" )
		print ("for example \n")
		print("python process_rosbag_clean.py ./rosbags/calibration_bag_transform5_2020-09-17-18-50-10.bag  ./rosbag_data/transf5 ")
		sys.exit()
	else: 
		filename = args[1]
		save_dir = args[2]

	create_directories(save_dir)
	process_rosbag (filename,save_dir)