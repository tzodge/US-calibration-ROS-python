# image number 1, 180 and 260 are used.
# Refer to image_1_calc.jpg for reference

import numpy as np
import matplotlib.pyplot as plt
import h5py
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

    

def find_scale_hor(pose_a,pose_b,image_b_coords,image_a_coords):

	# print ("del_pose =", pose_b - pose_a) 
	del_z = pose_a[2] - pose_b[2]
	del_px = (image_b_coords[:,0] - image_a_coords[:,0]).mean()

	scale_x = del_z*1e3/del_px

	print("Scale  hor = {} mm per pixel".format(abs(scale_x)) )
	return scale_x

def find_scale_ver(pose_a,pose_b,image_b_coords,image_a_coords):

	print ("del_pose =", pose_b - pose_a) 
	del_y = pose_a[1] - pose_b[1]
	del_py = (image_b_coords[:,1] - image_a_coords[:,1]).mean()
	print (del_y,"del_y")
	print (del_py,"del_py")
	scale_ver = del_y*1e3/del_py

	print("Scale  vert = {} mm per pixel".format(abs(scale_ver)) )
	return scale_ver


file_identifier = '2020-09-17-18-34-53'

image_sync_h5 = "./h5_files/image_"+file_identifier + "_sync.h5"
pose_sync_h5 = "./h5_files/pose_"+file_identifier + "_sync.h5"
pose_raw_h5 = "./h5_files/pose_"+file_identifier + ".h5"


f1 = h5py.File(image_sync_h5,'r')    
f2 = h5py.File(pose_sync_h5,'r')    
f3 = h5py.File(pose_raw_h5,'r')    

image_data = np.array(f1['image_data'])
pose_data = np.array(f2['pose_data'])
pose_data_raw = np.array(f3['pose_data'])
 

# np.where()

image_a_id = 5 # image numbers for horizontal scale  from  file_identifier = '2020-09-17-18-34-53'
image_b_id = 270 # image numbers for horizontal scale  from  file_identifier = '2020-09-17-18-34-53'

image_a_id = 625 # image numbers for horizontal scale  from  file_identifier = '2020-09-17-18-34-53'
image_b_id = 851 # image numbers for horizontal scale  from  file_identifier = '2020-09-17-18-34-53'


a = np.where(image_data[:,-1]==image_a_id) [0][0]# image numbers for horizontal scale  from  file_identifier = '2020-09-17-18-34-53'
b = np.where(image_data[:,-1]==image_b_id) [0][0]

print(a,"a")
print(b,"b")

## inserted manually using the code image_data.py
image_boundary = np.array([[0,0],
						   [720,0],
						   [720, 1280],
						   [0,1280]])

image_a_coords = np.loadtxt("file_" + file_identifier + "_img_" + str(image_a_id) + ".txt")

image_b_coords = np.loadtxt("file_" + file_identifier + "_img_" + str(image_b_id)+ ".txt")

pose_a = pose_data[a ,2:]
pose_b = pose_data[b,2:]
 
find_scale_hor(pose_a,pose_b,image_a_coords,image_b_coords)
find_scale_ver(pose_a,pose_b,image_a_coords,image_b_coords)
 
print(image_b_coords- image_a_coords)
'''
plt.scatter(image_a_coords[:,0],image_a_coords[:,1],s=500,label="image_a_coords")
plt.scatter(image_b_coords[:,0],image_b_coords[:,1],s=300,label="image_b_coords")
plt.scatter(image_boundary[:,1],image_boundary[:,0],label="image_boundary")

plt.legend()
plt.show()


plt.plot(range(len(pose_data)) , pose_data[:,2], label='x')
plt.plot(range(len(pose_data)) , pose_data[:,3], label='y')
plt.plot(range(len(pose_data)) , pose_data[:,4], label='z')
# plt.plot(range(len(pose_data)) , pose_data[:,5], label='rx')
# plt.plot(range(len(pose_data)) , pose_data[:,6], label='ry')
# plt.plot(range(len(pose_data)) , pose_data[:,7], label='rz')
# plt.plot(range(len(pose_data)) , pose_data[:,8], label='rw')
plt.legend()
plt.show()

'''
 
n = len(pose_data)
fig = plt.figure()
ax3D = fig.add_subplot(111, projection='3d')
ax3D.scatter(pose_data[:,2], pose_data[:,3], pose_data[:,4], s=10, c=np.arange(n), marker='o')  
ax3D.scatter(pose_data[a,2], pose_data[a,3], pose_data[a,4], s=100, c='red', marker='o')  
ax3D.scatter(pose_data[b,2], pose_data[b,3], pose_data[b,4], s=100, c='blue', marker='o')  
ax3D.set_xlabel("x axis")
ax3D.set_ylabel("y axis")
ax3D.set_zlabel("z axis")
ax3D.set_title('pose data synced')
plt.show()
 
'''

n = len(pose_data_raw)
fig = plt.figure()
ax3D = fig.add_subplot(111, projection='3d')
ax3D.scatter(pose_data_raw[:,3], pose_data_raw[:,4], pose_data_raw[:,5], s=10, c=np.arange(n), marker='o')  
# ax3D.scatter(pose_data_raw[a,2], pose_data_raw[a,3], pose_data_raw[a,4], s=100, c='red', marker='o')  
# ax3D.scatter(pose_data_raw[b,2], pose_data_raw[b,3], pose_data_raw[b,4], s=100, c='blue', marker='o')  
ax3D.set_xlabel("x axis")
ax3D.set_ylabel("y axis")
ax3D.set_zlabel("z axis")
ax3D.set_title('pose data raw')
plt.show()
 
'''
