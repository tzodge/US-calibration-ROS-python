# Standard imports
import cv2
import numpy as np
import sys

_ix,_iy=-1,-1

def find_centroids(im):
	kernel = np.ones((5,5),np.uint8)
	opening = cv2.morphologyEx(im, cv2.MORPH_OPEN, kernel)
	# ret,thresh1 = cv2.threshold(opening,127,255,cv2.THRESH_BINARY)
	ret,thresh1 = cv2.threshold(opening,80,255,cv2.THRESH_BINARY)


	# Find Canny edges 
	edged = cv2.Canny(thresh1, 30, 200) 
	# cv2.waitKey(0) 
	  

	contours, hierarchy = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
	  
	
	 
	centroid_list = []

	for contour in contours:
		# print(contour.shape,"contour.shape")

		color = np.random.rand(3,)*255

		centroid = (contour.mean(axis=0,dtype=np.int64)).flatten() 

		# cv2.drawContours(im, contour, -1, color=color, thickness=2) 

		centroid_list.append(centroid)

	return np.array(centroid_list)

def find_closest_point(centroids, point):
	dist = np.linalg.norm(centroids - np.array([point]) , axis=1)
	min_idx = np.argmin(dist)
	return centroids[min_idx,:]

def select_points(im):

	def on_click(event, x, y, p1, p2):
		global _ix, _iy
		if event == cv2.EVENT_LBUTTONDOWN:
			cv2.circle(im, (x, y), 3, (0, 0, 255), -1)
			_ix,_iy = x,y
			print (_ix,_iy,"ix,iy")
			print ("")
	  
	
	 
	centroid_array = find_centroids(im)

	for centroid in centroid_array:
		# print(contour.shape,"contour.shape")

		color = np.random.rand(3,)*255
		cv2.circle(im, (centroid[0],centroid[1]), 5, color=color,thickness=2)

	# cv2.imshow('Contours', im) 
	nn_old = [-1,-1]

	wire_pixels = []
	while(1):
		cv2.namedWindow('Contours')
		cv2.setMouseCallback('Contours', on_click)
		
		nn_new = find_closest_point(centroid_array, [_ix, _iy])
		# print(nn_new,"nn_new")
		im = cv2.rectangle(im,(nn_new[0]-3,nn_new[1]+3),(nn_new[0]+3,nn_new[1]-3),(0,255,0),3)
		if (nn_new != nn_old).any():
			nn_old = nn_new
			wire_pixels.append(nn_old)

		cv2.imshow('Contours',im)
		k = cv2.waitKey(20) & 0xFF
		if k == 27:
			break
		elif k == ord('a'):
			print mouseX,mouseY
	wire_pixel_np = np.array(wire_pixels[1:])
	print(wire_pixel_np,"wire_pixel_np")

	return wire_pixel_np


'''
image ids for transf2 80 102 125 177 255 321 363 390 490 664 738 800
'''
def update_selected_images(selected_images,available_images):
	
	### 
	### update iamge numbers to their nearest neighbours
	### in available images 	


	updated_images = np.zeros((len(selected_images)))
	for i in range(len(selected_images)):
		dist_array = abs(available_images - selected_images[i])
 
		nn_in_available = np.argmin(dist_array)
		updated_images[i] = available_images[nn_in_available]

	return updated_images.astype(np.int64)


if __name__ == '__main__':
	args = sys.argv
	if len(args) < 2:
		print( "\n Usage: python annotate_image_data_clean.py <rosbag_data_path>  " )
		sys.exit()
	save_dir = args[1]

	selected_ids_txt = save_dir + "/extra_files/selected_image_ids.txt"
	available_images_txt = save_dir + "/extra_files/available_images.txt"

	selected_images = np.loadtxt(selected_ids_txt, dtype=np.int64)
	available_images = np.loadtxt(available_images_txt, dtype=np.int64)

	if len(selected_images) < 1:
		print("please select images and put their respective numbers in \n")
		print(selected_ids_txt)
		sys.exit()
	print(selected_images,"selected_images before")
	# print(available_images,"available_images")
	selected_images = update_selected_images(selected_images,available_images)
	print(selected_images,"selected_images after")
	np.savetxt(selected_ids_txt,selected_images,fmt='%i')

	for image_number in selected_images:
		print (image_number,"image_number")

		im_path =  save_dir + "/calib_images" + "/image_{}.jpg".format(image_number)
		im = cv2.imread(im_path)
		wire_pixel_np = select_points(im)
		save_file = save_dir+ ""+"/image_frame_coords/"+"pix_coords_img_{}".format(image_number) + ".txt"
		np.savetxt( save_file,wire_pixel_np)
		cv2.destroyAllWindows() 

'''
'''

