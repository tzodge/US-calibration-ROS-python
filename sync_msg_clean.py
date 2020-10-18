import h5py
import numpy as np
import matplotlib.pyplot as plt
import time

class SearchSync(object):

	def __init__(self,file1_str,file2_str):
		
		self.time_f1 = []
		self.time_f2 = []
		self.t_thres = 0.05
		self.file1_str = file1_str
		self.file2_str = file2_str


	def binary_search(self,t,low,high):

		if high>low:

			mid = (low+high)//2

			if abs((self.time_f1[mid]-t))<self.t_thres:
				return mid
			elif (self.time_f1[mid]-t)>self.t_thres:
				# print("mid",mid)
				return self.binary_search(t,low,mid-1)
			elif (self.time_f1[mid]-t)<self.t_thres:
				# print("high",mid)
				return self.binary_search(t,mid+1,high)
		else:

			return 0

	def search_and_sync(self):
		start_time = time.time()
		low = 0
		self.count = 0
		self.f1_synced_array = []
		self.f2_synced_array = []
		self.f1_synced_ = []
		self.f2_synced_ = []
		high = len(self.time_f1)
		for i in range(len(self.time_f2)):
			j = self.binary_search(self.time_f2[i],low,high)
			if j>0:
				self.f1_synced_array.append(self.time_f1[j]) #redundant
				self.f2_synced_array.append(self.time_f2[i]) #redundant

				self.f1_synced_.append(np.array([self.f1_values[j,1],self.f1_values[j,2],self.f1_values[j,3],self.f1_values[j,4],self.f1_values[j,5],
												self.f1_values[j,6],self.f1_values[j,7],self.f1_values[j,8],self.f1_values[j,9]]).reshape(1,-1)) #creating synced micron file with coordinate values
				self.f2_synced_.append(np.array([self.f2_values[i,1],self.f2_values[i,2],i]).reshape(1,-1))

				print("matching pairs",i, self.f1_synced_array[self.count],self.f2_synced_array[self.count],self.time_f2[i]-self.time_f2[0])
				# print([self.f2_values[i,1],self.f2_values[i,2],i])
				self.count += 1

		# print(self.synced_array[0])
		print("total time to sync: ",time.time()-start_time)
		# self.plotting()

		pass

	def save_h5(self, f1_synced, f2_synced):
 
		# self.f1_synced = h5py.File("."+self.file1_str.split('.')[1] + '_sync' + '.h5' ,'w')
		# self.f2_synced = h5py.File("."+self.file2_str.split('.')[1] + '_sync' + '.h5' ,'w')
		self.f1_synced = h5py.File(f1_synced ,'w')
		self.f2_synced = h5py.File(f2_synced ,'w')


		print("."+self.file1_str.split('.')[1] + '_sync' + '.h5' )

		self.f1_synced_ = np.concatenate(self.f1_synced_)
		self.f2_synced_ = np.concatenate(self.f2_synced_)

		array_aft = np.array(self.f1_synced_)

		self.f1_synced.create_dataset("pose_data",data=self.f1_synced_)
		self.f2_synced.create_dataset("image_data",data=self.f2_synced_)
 

	def plotting(self):
		plt.title("Time synced pose_data nd image, delta_t=0.05")
		plt.xlabel('image data timestamp')
		plt.ylabel('pose_data data timestamp')
		plt.scatter(self.f2_synced_array,self.f1_synced_array,c=(0,0,0),alpha=0.5)
		plt.show()

	def read_msgs(self):

		# self.f1 = h5py.File('micron_full_leg_1.h5','r')
		# self.f2 = h5py.File('image_data.h5','r')

		self.f1 = h5py.File(self.file1_str,'r')
		self.f2 = h5py.File(self.file2_str,'r')

		self.f1_values = self.f1['pose_data']
		self.f2_values = self.f2['image_data']


		for i in range(self.f2_values.shape[0]):
			self.time_f2.append(self.f2_values[i,1]+1e-9*self.f2_values[i,2])
		for j in range(self.f1_values.shape[0]):
			self.time_f1.append(self.f1_values[j,1]+1e-9*self.f1_values[j,2])

		# print(self.time_f1,"self.time_f1")
		# print(self.time_f2,"self.time_f2")
		# plt.figure()
		# plt.plot(np.arange(len(self.time_f1)), self.time_f1,label = 'pose')
		# plt.plot(np.arange(len(self.time_f2)), self.time_f2,label = 'image')
		# plt.legend()
		# plt.show()
		pass

if __name__ == "__main__":
	# file_identifier = "2020-09-01-16-32-36"
	# file_identifier = "2020-09-01-16-33-50"
	# file_identifier = "2020-09-01-16-40-34"
	# file_identifier = '2020-09-01-16-47-46'
	file_identifier = '2020-09-17-18-34-53'
	f_pose = "./h5_files/pose_"+ file_identifier +".h5" 
	f_image = "./h5_files/image_"+ file_identifier +".h5" 

	f_pose_sync = "./h5_files/pose_"+ file_identifier+"_sync" +".h5" 
	f_image_sync = "./h5_files/image_"+ file_identifier+"_sync" +".h5" 
	

	ss = SearchSync(f_pose,f_image)
	ss.read_msgs()	#micron_coordinates,image_pixel_coordinates
	ss.search_and_sync()
	ss.save_h5(f_pose_sync,f_image_sync)
	
	# sorted_l = search_and_sync(l1,l2)
