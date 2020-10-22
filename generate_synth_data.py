import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import transforms3d as t3d
import csv



def transformation_from_R_t(R,t):
    M = np.eye(4)
    M[0:3,0:3] = R
    M[0:3,3] = t
    return M


def generate_rand_transf(axis=None, 
                        angle=None, 
                        translation=None):
    if axis == None:
        axis = np.random.rand(3,) - np.array([0.5,0.5,0.5])
        axis = axis/np.linalg.norm(axis)
    if angle == None:
        angle = 2*(np.random.uniform()-0.5) * np.pi  
    if translation == None:
        translation = np.random.rand(3,)

    R = t3d.axangles.axangle2mat(axis, angle)     

    M = transformation_from_R_t(R,translation)

    return M, axis, angle, translation

def generate_n_transf(n, method="random"):
    Transf_list = []

    for i in range(n):
        if method =="random":
            Transf_list.append(generate_rand_transf()[0])
        elif method =="structured":
            axis = [ 1,0,0]
            # axis = [1- np.random.rand()*0.1,1- np.random.rand()*0.1,1- np.random.rand()*0.1] 
            angle =  np.random.rand()
            # angle = np.pi*i/n *1/3
            translation = [i*0.05,0,0]
            transf = generate_rand_transf(axis=axis,
                                         angle=angle,
                                         translation=translation)[0]
            Transf_list.append(transf)

    Transf_array = np.array(Transf_list)


    return  Transf_array

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

def plot_base_frame(ax_handle):
    ax_handle.scatter(0, 0, 0, s=10,  marker='o')  
    ax_handle.text(0, 0, 0, 'Robot base')   
    plot_frame(np.eye(4),ax_handle)

def plot_frames(frames,ax_handle, prefix = "_"):

    for i,frame in zip(range(len(frames)),frames): 
    # for frame in frames: 
 
        plot_frame(frame,ax_handle) 
        np.set_printoptions(precision=2)

 
        # ax_handle.scatter(frame_orig[0], frame_orig[1], frame_orig[2], s=100,  marker='o')  
        ax_handle.text(frame[0,3], frame[1,3], frame[2,3], prefix+'_{}'.format(i))     
        ax_handle.set_xlabel('x_axis')
        ax_handle.set_ylabel('y_axis')
        ax_handle.set_zlabel('z_axis')

    # ax_handle.set_aspect('equal')


def write_to_csv(csv_file_name, Transf_array):
    csv_file = open(csv_file_name, 'w')
    for i in range(len(Transf_array)):
        tf_mat = Transf_array[i]
        translation =tf_mat[0:3,3]
        R = tf_mat[0:3,0:3]
        qw,qx,qy,qz = t3d.quaternions.mat2quat(R) 

        csv_file.write(
                    str(i) + ', ' +
                    str(translation[0]) + ', ' + str(translation[1]) + ', ' +
                    str(translation[2]) + ', ' + str(qx) + ', ' +
                    str(qy) + ', ' + str(qz) + ', ' +
                    str(qw) + '\n')



n = 10

Pb = np.random.rand(4,1)
Pb[3,0] = 1

Pz = np.random.rand(4,1)
Pz[3,0] = 1

T_ue_gt = generate_rand_transf()[0]  ## End effector orientation in US frame 
# T_zu_array = generate_n_transf(n,"structured")
T_zu_array = generate_n_transf(n)

# print (T_zu_array,"T_zu_array")

T_zb_gt = generate_rand_transf()[0]  ## Base frame in phantom i.e. world frame

T_be_list = []

for i in range(n):
    term1 = np.linalg.inv(T_zb_gt)
    term2 = T_zu_array[i]
    term3 = T_ue_gt

    T_be_list.append(term1.dot(term2).dot(term3))    

T_be_array = np.array(T_be_list)

fig = plt.figure()
ax_handle = fig.add_subplot(111, projection="3d")

write_to_csv(csv_file_name = "temp_quat_csv/T_be_array.csv", Transf_array=T_be_array)
write_to_csv(csv_file_name = "temp_quat_csv/T_zu_array.csv", Transf_array=T_zu_array)
write_to_csv(csv_file_name = "temp_quat_csv/T_ue_gt.csv", Transf_array=[T_ue_gt])
write_to_csv(csv_file_name = "temp_quat_csv/T_zb_gt.csv", Transf_array=[T_zb_gt])
write_to_csv(csv_file_name = "temp_quat_csv/T_ue_gt_inv.csv", Transf_array=[np.linalg.inv(T_ue_gt)])
write_to_csv(csv_file_name = "temp_quat_csv/T_zb_gt_inv.csv", Transf_array=[np.linalg.inv(T_zb_gt)])

plot_base_frame(ax_handle)
plot_frames(T_be_array,ax_handle,prefix="__EE")
plot_frames(T_be_array,ax_handle,prefix="__EE")



ax_handle.set_aspect("equal")
ax_handle.set_xlim([-0.5,0.5])
ax_handle.set_ylim([-0.5,0.5])
ax_handle.set_zlim([-0.5,0.5])
plt.show()


'''
this code will save generated transformation csv to temp_csv

copy those files to /home/biorobotics/catkin_ws/src/hand_eye_calibration/hand_eye_calibration/bin/temp_quat_csv

Then run 
python compute_hand_eye_calibration.py \
--aligned_poses_B_H_csv_file temp_quat_csv/T_be_array.csv \
--aligned_poses_W_E_csv_file temp_quat_csv/T_zu_array.csv \
--visualize=True \
--plot_every_nth_pose 1

Note that the convention for a variable denoted in eth's work is reverse from ours

so the out pose_H_E should be same as T_ue_gt_inv
'''