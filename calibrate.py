
import numpy as np
import matplotlib.pyplot as plt
import h5py
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import sys
from frame_from_coords_clean import *
import transforms3d as t3d
from register import register

def create_transf_mat(R,t):
    T = np.zeros((4,4))
    T[3,3] = 1
    T[0:3,0:3] = R
    T[0:3,3] = t
    return T

def pose_vect_to_transf(pose_vect):
    # print(pose_vect,"pose_vect")
    q = np.array([pose_vect[6],pose_vect[3],pose_vect[4],pose_vect[5]])
    R = t3d.quaternions.quat2mat(q)
    t = pose_vect[0:3]
    T = create_transf_mat(R,t)
    return T

def pose_for_image_id(pose_data, image_data,image_id):
    pose_idx = np.where(image_data[:,-1]== image_id) [0][0]
    pose_vect = pose_data[pose_idx,2:]
    
    T = pose_vect_to_transf(pose_vect)
    return T

def extract_frames(save_dir):

    image_ids = np.loadtxt(save_dir + "/extra_files/selected_image_ids.txt")
    T_zu_array = np.load(save_dir + "/extra_files/ultrasound_frames.npz")['ultrasound_frames']
    pose_sync_h5 = save_dir + "/h5_files/pose_data_sync.h5"
    image_sync_h5 = save_dir + "/h5_files/image_data_sync.h5"


    f1 = h5py.File(image_sync_h5,'r')    
    f2 = h5py.File(pose_sync_h5,'r')    
    # f3 = h5py.File(pose_raw_h5,'r')    

    image_data = np.array(f1['image_data'])
    pose_data = np.array(f2['pose_data'])


    T_be_list = []
    for image_id in image_ids:
        T_be = pose_for_image_id(pose_data, image_data,image_id)
        T_be_list.append(T_be)
    T_be_array = np.array(T_be_list)

    return T_be_array, T_zu_array


def plot_frame(frame, ax_handle):
    R = frame[0:3,0:3] 
    frame_orig = frame[0:3,0:3] 
    x_axis,y_axis,z_axis = R[:,0], R[:,1], R[:,2]   
    frame_orig = frame[0:3,3]
    # c = np.random.rand(3,)
    s = 0.1

    ax_handle.quiver(frame_orig[0], frame_orig[1], frame_orig[2], \
                s * x_axis[0], \
                s * x_axis[1], \
                s * x_axis[2], color='r') # plotting x_axis  
    ax_handle.quiver(frame_orig[0], frame_orig[1], frame_orig[2], \
                s * y_axis[0], \
                s * y_axis[1], \
                s * y_axis[2], color='g') # plotting x_axis  
    ax_handle.quiver(frame_orig[0], frame_orig[1], frame_orig[2], \
                s * z_axis[0], \
                s * z_axis[1], \
                s * z_axis[2], color='b') # plotting x_axis  



def plot_frames(frames,ax_handle):


    ax_handle.scatter(0, 0, 0, s=10,  marker='o')  
    ax_handle.text(0, 0, 0, 'Robot base')   
    plot_frame(np.eye(4),ax_handle)
    for i,frame in zip(range(len(frames)),frames): 
    # for frame in frames: 
 
        plot_frame(frame,ax_handle) 
        np.set_printoptions(precision=2)

 
        # ax_handle.scatter(frame_orig[0], frame_orig[1], frame_orig[2], s=100,  marker='o')  
        ax_handle.text(frame[0,3], frame[1,3], frame[2,3], ' {}'.format(i))     
        ax_handle.set_xlabel('x_axis')
        ax_handle.set_ylabel('y_axis')
        ax_handle.set_zlabel('z_axis')

    # ax_handle.set_aspect('equal')



def invert_transf_array(T_list):
    T_inv_list = []
    
    for T in T_list:
        T_inv_list.append(np.linalg.inv(T))
    
    return np.array(T_inv_list)

if __name__ == '__main__':
    args = sys.argv
    if len(args) != 2:
        print( "\n Usage: python frame_from_coords_clean.py <rosbag_data_path> <image_number>" )
        sys.exit()
    save_dir = args[1]
    # save_dirs = ["./rosbag_data/transf1/","./rosbag_data/transf2/",\
    #            "./rosbag_data/transf3/","./rosbag_data/transf4/",\
    #            "./rosbag_data/transf5/"]
 
    # save_dirs = ["./rosbag_data/transf2/", "./rosbag_data/transf1/" ]
    save_dirs = ["./rosbag_data/transf2/" ]


    T_be_array = np.empty((1,4,4))  
    T_zu_array = np.empty((1,4,4))  
    for save_dir in save_dirs: 
        T_be_array_curr, T_zu_array_curr = extract_frames(save_dir)
        # print (T_be_array_curr.shape,"T_be_array_curr.shape")
        T_be_array = np.vstack((T_be_array, T_be_array_curr))
        T_zu_array = np.vstack((T_zu_array, T_zu_array_curr))
     
    T_be_array = T_be_array[1:,:,:]
    T_zu_array = T_zu_array[1:,:,:]

    T_eb_array = invert_transf_array(T_be_array)

    T_uz_array = invert_transf_array(T_zu_array)
    # R_est = np.array([[ 0, 1, 0],
    #                 [ 0, 0,-1],
    #                 [-1, 0, 0]])

    # R_est = np.array([[ 1, 0, 0],
    #                   [ 0, 0,-1],
    #                   [ 0,1, 0]])
    eul_out_rob = np.zeros((len(T_eb_array),3))
    eul_out_camera = np.zeros((len(T_eb_array),3))

    for i in range(len(T_eb_array)):
        eul_out_rob[i,:] =  t3d.euler.mat2euler(T_eb_array[i,0:3,0:3])
        print(np.array(eul_out_rob)*180/np.pi,"eul_out_rob deg, robot")

        eul_out_camera[i,:] =  t3d.euler.mat2euler(T_zu_array[i,0:3,0:3])
        print(np.array(eul_out_camera)*180/np.pi,"eul_out_camera deg, US camera frame")

    eul_out_camera = eul_out_camera*180/np.pi
    eul_out_rob = eul_out_rob*180/np.pi
    fig = plt.figure()
    ax_handle = fig.add_subplot(111, projection='3d')
    ax_handle.scatter(eul_out_rob[:,0], eul_out_rob[:,0], eul_out_rob[:,0], label="robot eul")
    ax_handle.scatter(eul_out_camera[:,0], eul_out_camera[:,0], eul_out_camera[:,0], label="camera eul")
    plt.legend()
    plt.show()


    R_est = np.array([[ -1, 0, 0],
                      [ 0, 0, 1],
                      [ 0,1, 0]])


    # R_est = t3d.axangles.axangle2mat(np.random.rand(3,),np.random.rand())

    # t_est = np.array([ -0.2, 0.0, 0.15])
    # t_est = np.array([ 0.2, 0.0, 0.15])
    t_est = np.array([ 0.12, 0.0, 0.0443])
    # t_est = np.random.rand(3,)*0.10
    T_eu_est = create_transf_mat(R_est,t_est)

    # P_z = np.array([0,0,0,1]).T
    P_z = np.random.rand(4)*0.01   ##  a random point fixed in the z wire frame
    P_z[3] = 1
    

    fig = plt.figure()
    ax_handle = fig.add_subplot(111, projection='3d')
    plot_frames(T_be_array ,ax_handle)
    plot_frames([T_be_array[0]],ax_handle)

    n_itr = 1000
    for i in range(n_itr):
        # fig = plt.figure()
        # ax_handle = fig.add_subplot(111, projection='3d')

        np.set_printoptions(precision=3)
         
        P_const_est_list = []
        for i in range(len(T_zu_array)):
            P_const_est_i = T_be_array[i].dot(T_eu_est).dot(T_uz_array[i]).dot(P_z)
            P_const_est_list.append(P_const_est_i)

        P_const_est_array = np.array(P_const_est_list)  
        P_const_est = P_const_est_array.mean(axis=0)
        # P_const_est = P_const_est_array[0]
        np.set_printoptions(precision=5)
        # print (P_const_est_array,"P_const_est_array")
        # print (P_const_est,"P_const_est")

        P_const_est = T_be_array[0].dot(T_eu_est).dot(T_uz_array[0]).dot(P_z)

        P_h_list = []
        for i in range(len(T_uz_array)):
            P_h_list.append(T_uz_array[i].dot(P_z))
        
        Q_h_list = []
        for i in range(len(T_eb_array)):
            Q_h_list.append(T_eb_array[i].dot(P_const_est))

        P_h_array = np.array(P_h_list)  
        Q_h_array = np.array(Q_h_list)
        
        # print(P_h_array,"P_h_array")
        # print(Q_h_array,"Q_h_array")

        R_eu,t_eu = register(P_h_array[:,0:3], Q_h_array[:,0:3])
        # ax_handle.scatter(P_h_array[:,0], P_h_array[:,1], P_h_array[:,2],  label='P')
        # ax_handle.scatter(Q_h_array[:,0], Q_h_array[:,1], Q_h_array[:,2],  label='Q')
        # ax_handle.quiver(P_h_array[:,0], P_h_array[:,1], P_h_array[:,2] , \
        #                  Q_h_array[:,0], Q_h_array[:,1], Q_h_array[:,2] )
'''

        # for itr in range(len(P_h_array)):
        #     ax_handle.plot([P_h_array[itr,0], Q_h_array[itr,0]],\
        #                    [P_h_array[itr,1], Q_h_array[itr,1]],\
        #                    [P_h_array[itr,2] ,Q_h_array[itr,2]] , c='r', alpha=0.5)

        
        T_eu_est = T_eu_est.dot(create_transf_mat(R_eu,t_eu))
        T_eu_est = create_transf_mat(R_eu,t_eu)

        print(T_eu_est,"T_eu_est")
        P_transf = (T_eu_est.dot(P_h_array.T)).T

        # ax_handle.scatter(P_transf[:,0], P_transf[:,1], P_transf[:,2],  label='P_transf')
        # ax_handle.set_aspect('equal')

        # plt.legend()
        # plt.show()
        
        for i in range(len(T_be_array)):
            P_const = T_be_array[i].dot(T_eu_est).dot(T_uz_array[i]).dot(P_z)
            ax_handle.scatter(P_const_est[0],P_const_est[1],P_const_est[2], s=100) 
            plot_frame(T_be_array[0].dot(T_eu_est),ax_handle)

        plt.show()

    # P_const_new = p
    # print (P_const,"P_const")

'''
