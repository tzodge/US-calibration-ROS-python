# Standard imports
import cv2
import numpy as np
import sys

ix,iy=-1,-1
def select_points(save_dir,image_number):

	def on_click(event, x, y, p1, p2):
		global ix, iy
		if event == cv2.EVENT_LBUTTONDOWN:
			cv2.circle(im, (x, y), 3, (0, 0, 255), -1)
			ix,iy = x,y
			print (ix,iy,"ix,iy")
			print ("")
 
 
	# im = cv2.imread("./calib_images_" + file_identifier+ "/image_"+ str(img_number) +".jpg",cv2.COLOR_BGR2GRAY)
	im_path =  save_dir + "/calib_images" + "/image_{}.jpg".format(image_number)
	im = cv2.imread(im_path)
	gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) 
	 

	kernel = np.ones((5,5),np.uint8)
	opening = cv2.morphologyEx(im, cv2.MORPH_OPEN, kernel)
	# ret,thresh1 = cv2.threshold(opening,127,255,cv2.THRESH_BINARY)
	ret,thresh1 = cv2.threshold(opening,80,255,cv2.THRESH_BINARY)


	# Find Canny edges 
	edged = cv2.Canny(thresh1, 30, 200) 
	cv2.waitKey(0) 
	  

	_, contours, hierarchy = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
	  
	
	 
	centroid_list = []

	for contour in contours:
		# print(contour.shape,"contour.shape")

		color = np.random.rand(3,)*255

		centroid = (contour.mean(axis=0,dtype=np.int64)).flatten() 

		# cv2.drawContours(im, contour, -1, color=color, thickness=2) 

		centroid_list.append(centroid)
		cv2.circle(im, (centroid[0],centroid[1]), 5, color=color,thickness=2)

	centroid_array = np.array(centroid_list)
	# cv2.imshow('Contours', im) 
	nn_old = [-1,-1]

	wire_pixels = []
	while(1):
		cv2.namedWindow('Contours')
		cv2.setMouseCallback('Contours', on_click)
		dist = np.linalg.norm(centroid_array - np.array([[ix,iy]]) , axis=1)
		

		min_idx = np.argmin(dist)
		nn_new = centroid_array[min_idx,:]
		print(nn_new,"nn_new")
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

	save_file = save_dir+ ""+"/image_frame_coords/"+"pix_coords_img_{}".format(image_number) + ".txt"
	
	np.savetxt( save_file,wire_pixel_np)
	cv2.destroyAllWindows() 


'''
image ids for transf2 80 102 125 177 255 321 363 390 490 664 738 800
'''

if __name__ == '__main__':
	args = sys.argv
	if len(args) < 2:
		print( "\n Usage: python annotate_image_data_clean.py <rosbag_data_path>  " )
		sys.exit()
	save_dir = args[1]

	if len(args) == 3:
		selected_ids_txt = args[2]
		image_numbers = np.loadtxt( selected_ids_txt , dtype=np.int64)
	else:	
		selected_ids_txt = save_dir + "/extra_files/selected_image_ids.txt"
 		image_numbers = np.loadtxt(selected_ids_txt, dtype=np.int64)

	 
	for image_number in image_numbers:
		print (image_number,"image_number")
		select_points(save_dir,image_number)


