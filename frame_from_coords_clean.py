
import numpy as np
import matplotlib.pyplot as plt
import h5py
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import sys
import transforms3d as  t3d
# link : https://docs.google.com/presentation/d/1E5cLrZCGC0eLe_q-Whda-CIxITOYS3GRCLCoh0wb2Lg/edit?usp=sharing

def centralize_and_scale(image_coords,img_origin):
	scale_ver = 0.0953107555263 * 1e-3  # m/pix
	scale_hor = 0.074094097377  * 1e-3  # m/pix

	image_coords_cent =  image_coords - img_origin 
	image_coords_cent[:,0] = scale_hor*image_coords_cent[:,0]
	image_coords_cent[:,1] = scale_ver*image_coords_cent[:,1]

	return image_coords_cent


def create_transf_mat(R,t):
	T = np.zeros((4,4))
	T[3,3] = 1
	T[0:3,0:3] = R
	T[0:3,3] = t
	return T


def img_3d_coords_zFrame (image_coords_cent ): 
	

	 
	p1 = image_coords_cent[0,:]
	p2 = image_coords_cent[1,:]
	p3 = image_coords_cent[2,:]
	p4 = image_coords_cent[3,:]
	p5 = image_coords_cent[4,:]
	p6 = image_coords_cent[5,:]
	p7 = image_coords_cent[6,:]
	p8 = image_coords_cent[7,:]
	p9 = image_coords_cent[8,:]


	 
	P0 = np.array([0,0,0])
	## calculate 3D location of P2
	r2 = np.linalg.norm(p2-p1) / np.linalg.norm(p3-p1)
	P2_x = 0.08*(r2-0.25) 
	P2_y = 0.005 + 0.02*(r2-0.25) 
	P2_z = 0.0

	P2 = np.array([P2_x,P2_y,P2_z])  # center point first line

	## calculate 3D location of P5
	r5 = np.linalg.norm(p5-p4) / np.linalg.norm(p6-p4)
	P5_x = -0.08*(r5-0.75) 
	P5_y = 0.015 + 0.02*(r5-0.75) 
	P5_z = -0.010

	P5 = np.array([P5_x,P5_y,P5_z])  # center point first line


	## calculate 3D location of P5
	r8 = np.linalg.norm(p8-p7) / np.linalg.norm(p9-p7)
	P8_x =  0.04*(3*r8-1) 
	P8_y = 0.005 + 0.005*(3*r8-1) 
	P8_z = -0.020

	P8 = np.array([P8_x,P8_y,P8_z])  # center point first line


	## calculate 3D location of P1
	P1_y = 0.0
	P1_z = 0.0
	Rad1 = np.linalg.norm(p2-p1)

 	#####
 	## Explaination
 	#####
	if P2_x > P5_x: 
		P1_x = P2_x + np.sqrt(Rad1**2 - P2_y**2) # GIVE NAME TO THE SQRT FUNC
	else:
		P1_x = P2_x - np.sqrt(Rad1**2 - P2_y**2)

	P1 = np.array([P1_x,P1_y,P1_z])


	## calculate 3D location of P3
	P3_y = 0.020
	P3_z = 0.0
	Rad3 = np.linalg.norm(p2-p3)

	if P2_x > P5_x: 
		P3_x = P2_x - np.sqrt(abs(Rad3**2 - (0.02-P2_y)**2))
	else:
		P3_x = P2_x + np.sqrt(abs(Rad3**2 - (0.02-P2_y)**2))

	P3 = np.array([P3_x,P3_y,P3_z])





	## calculate 3D location of P1
	P4_y = 0.0
	P4_z = -0.010
	Rad4 = np.linalg.norm(p5-p4)
	if P2_x > P5_x: 
		P4_x = P5_x + np.sqrt(Rad4**2 - P5_y**2)
	else:
		P4_x = P5_x - np.sqrt(Rad4**2 - P5_y**2)

	P4 = np.array([P4_x,P4_y,P4_z])


	P6_y = 0.020
	P6_z = -0.010
	Rad6 = np.linalg.norm(p5-p6)
	if P2_x > P5_x: 
		P6_x = P5_x - np.sqrt(Rad6**2 - (0.02-P5_y)**2)
	else:
		P6_x = P5_x + np.sqrt(Rad6**2 - (0.02-P5_y)**2)

	P6 = np.array([P6_x,P6_y,P6_z])



	P3D = np.zeros((7,3))

	P3D[0,:] = P0  ## [0,0,0]
	P3D[1,:] = P1
	P3D[2,:] = P2
	P3D[3,:] = P3
	P3D[4,:] = P4
	P3D[5,:] = P5
	P3D[6,:] = P6

	return P3D 
def angle_between_vec(v1,v2):
	v1_u = v1/np.linalg.norm(v1)
	v2_u = v2/np.linalg.norm(v2)
	theta = np.arccos(v1_u.dot(v2_u))
	return theta


def find_img_frame_orig(image_coords_cent,P3D):	
	p1 = image_coords_cent[0,:]
	p2 = image_coords_cent[1,:]
	p3 = image_coords_cent[2,:]
	p4 = image_coords_cent[3,:]
	p5 = image_coords_cent[4,:]
	p6 = image_coords_cent[5,:]
	p7 = image_coords_cent[6,:]
	p8 = image_coords_cent[7,:]
	p9 = image_coords_cent[8,:]


	P0 = P3D[0,:] 
	P1 = P3D[1,:] 
	P2 = P3D[2,:] 
	P3 = P3D[3,:] 
	P4 = P3D[4,:] 
	P5 = P3D[5,:] 
	P6 = P3D[6,:] 

	theta =  angle_between_vec(p1, p2-p1)
	l_p1 = np.linalg.norm(p1)

	img_frame_orig = P1 - l_p1*np.cos(theta)* (P2-P1)/np.linalg.norm(P2-P1) \
						- l_p1*np.sin(theta)* (P4-P1)/np.linalg.norm(P4-P1) 

	return img_frame_orig

def refine_R(R):
	U,S,Vt = np.linalg.svd(R)
	# R = U.dot(Vt.T)
	return  U.dot(Vt)

def find_img_frame_axis(image_coords_cent,P3D):	
	p1 = image_coords_cent[0,:]
	p2 = image_coords_cent[1,:]
	p3 = image_coords_cent[2,:]
	p4 = image_coords_cent[3,:]
	p5 = image_coords_cent[4,:]
	p6 = image_coords_cent[5,:]
	p7 = image_coords_cent[6,:]
	p8 = image_coords_cent[7,:]
	p9 = image_coords_cent[8,:]


	P0 = P3D[0,:] 
	P1 = P3D[1,:] 
	P2 = P3D[2,:] 
	P3 = P3D[3,:] 
	P4 = P3D[4,:] 
	P5 = P3D[5,:] 
	P6 = P3D[6,:] 

	theta =  -angle_between_vec( p3-p1 , np.array([1,0]))
	x_axis = np.cos(theta)* (P3-P1)/np.linalg.norm(P3-P1) \
				+ np.sin(theta) * (P1-P4)/np.linalg.norm(P1-P4)  ## explain 

 	P3_cent = P3D[1:,:] - np.mean(P3D[1:,:], axis=0)
	U,S,Vt =  np.linalg.svd(P3_cent)
	V = Vt.T
	z_axis = V[:,-1]
	z_dir_vec = np.cross(P3-P1,P4-P1 )


 	if z_axis.dot(z_dir_vec) < 0:
 		z_axis = -1*z_axis

	y_axis = np.cross(z_axis,x_axis) ## calculate with geometry

	# return img_frame_orig
	R = np.zeros((3,3))
	R[:,0] = x_axis
	R[:,1] = y_axis
	R[:,2] = z_axis

	R_refined = refine_R(R)
	return R_refined


# def find_transf(image_coords,img_origin):
# 	image_coords_cent = centralize_and_scale(image_coords,img_origin)
# 	P3D = img_3d_coords_zFrame (image_coords_cent)
# 	img_frame_orig = find_img_frame_orig(image_coords_cent,P3D)
# 	R = find_img_frame_axis(image_coords,P3D)

# 	return R,img_frame_orig

'''
image ids for transf2 80 102 125 177 255 321 363 390 490 664 738 800
'''
if __name__ == '__main__':
	# args = sys.argv
	# if len(args) < 3:
	# 	print( "\n Usage: python frame_from_coords_clean.py <rosbag_data_path> <image_number>" )
	# 	sys.exit()
	# save_dir = args[1]
	# image_ids = args[2:]
	 
	
	# image_ids = [5,270,625,800]

	img_origin = np.array([388,134])
	args = sys.argv
	if len(args) < 2:
		print( "\n Usage: python annotate_image_data_clean.py <rosbag_data_path> <image_number 1> <image_number 2> " )
		sys.exit()
	save_dir = args[1]

	if len(args) == 3:
		selected_ids_txt = args[2]
		image_numbers = np.loadtxt( selected_ids_txt , dtype=np.int64)
	else:	
		selected_ids_txt = save_dir + "/extra_files/selected_image_ids.txt"
 		image_numbers = np.loadtxt(selected_ids_txt, dtype=np.int64)


 	print (image_numbers)
	fig = plt.figure()
	ax3D = fig.add_subplot(111, projection='3d')
	ultrasound_frames_list = []

	for image_id in image_numbers: 
		print(image_id,"image_id")
 
		pix_coords_path = save_dir + "/image_frame_coords/pix_coords_img_{}.txt".format(image_id)
		image_coords = np.loadtxt(pix_coords_path)
		
		# R,img_frame_orig = find_transf(image_coords,img_origin)
		image_coords_cent = centralize_and_scale(image_coords,img_origin)

		## temp
		# print (image_coords_cent,"image_coords_cent")
		# plt.scatter(image_coords_cent[:,0], image_coords_cent[:,1])
		# plt.show()
		## temp


		P3D = img_3d_coords_zFrame (image_coords_cent)

 		img_frame_orig = find_img_frame_orig(image_coords_cent,P3D)
		R = find_img_frame_axis(image_coords,P3D)
		eul_out =  t3d.euler.mat2euler(R)
		print(np.array(eul_out)*180/np.pi,"eul_out deg")
		ultrasound_frames_list.append( create_transf_mat(R,img_frame_orig))
		x_axis,y_axis,z_axis = R[:,0], R[:,1], R[:,2]	

		c = np.random.rand(3,)
		for i in range(len(P3D)):
			ax3D.scatter(P3D[i,0], P3D[i,1], P3D[i,2], s=10,  marker='o', c=c)  
			# ax3D.text(P3D[i,0], P3D[i,1], P3D[i,2], 'P{}'.format(i))  	

		ax3D.quiver(img_frame_orig[0], img_frame_orig[1], img_frame_orig[2], \
				    0.01 * x_axis[0], \
				    0.01 * x_axis[1], \
				    0.01 * x_axis[2]) # plotting x_axis  
		ax3D.quiver(img_frame_orig[0], img_frame_orig[1], img_frame_orig[2], \
				    0.01 * y_axis[0], \
				    0.01 * y_axis[1], \
				    0.01 * y_axis[2]) # plotting x_axis  
		ax3D.quiver(img_frame_orig[0], img_frame_orig[1], img_frame_orig[2], \
				    0.01 * z_axis[0], \
				    0.01 * z_axis[1], \
				    0.01 * z_axis[2]) # plotting x_axis  


		ax3D.scatter(img_frame_orig[0], img_frame_orig[1], img_frame_orig[2], s=100,  marker='o', c=c)  
		ax3D.text(img_frame_orig[0], img_frame_orig[1], img_frame_orig[2], ' {}'.format(image_id))  	
		ax3D.set_xlabel('x_axis')
		ax3D.set_ylabel('y_axis')
		ax3D.set_zlabel('z_axis')

	ultrasound_frames_file = save_dir + "/extra_files/ultrasound_frames.npz"
	np.set_printoptions(precision=3)
	print(np.array(ultrasound_frames_list),"ultrasound_frames_list")
	np.savez(ultrasound_frames_file, ultrasound_frames=np.array(ultrasound_frames_list))

	ax3D.set_aspect('equal')
	plt.show()

		