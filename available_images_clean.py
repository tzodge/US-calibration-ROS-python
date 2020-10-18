import numpy as np
import matplotlib.pyplot as plt
import h5py
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import sys


# file_identifier = '2020-09-17-18-34-53'

# image_sync_h5 = "./h5_files/image_"+file_identifier + "_sync.h5"
# pose_sync_h5 = "./h5_files/pose_"+file_identifier + "_sync.h5"
# pose_raw_h5 = "./h5_files/pose_"+file_identifier + ".h5"

def available_images(save_dir, plotting=False):
	
	image_sync_h5 = save_dir + "/h5_files/image_data_sync.h5"
	pose_sync_h5 = save_dir + "/h5_files/pose_data_sync.h5"
	pose_raw_h5 = save_dir + "/h5_files/pose_data.h5"


	f1 = h5py.File(image_sync_h5,'r')    
	f2 = h5py.File(pose_sync_h5,'r')    
	f3 = h5py.File(pose_raw_h5,'r')    

	image_data = np.array(f1['image_data'])
	pose_data = np.array(f2['pose_data'])
	pose_data_raw = np.array(f3['pose_data'])

	txt_file = save_dir + "/extra_files/available_images.txt"
	np.savetxt(txt_file, image_data[:,-1] , fmt='%i')

	if plotting:
		plt.scatter( image_data[:,-1], np.ones(len(image_data)) )
		plt.show()


if __name__ == '__main__':
	args = sys.argv
	if len(args) != 2:
		print( "\n Usage: python available_images_clean.py <rosbag_data_path> " )
		sys.exit()
	save_dir = args[1]
	#  save_dir = ./rosbag_data/scale_data
	available_images(save_dir, plotting=True)