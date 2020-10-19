from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import h5py
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import sys
import transforms3d as  t3d

def centralize_and_scale(image_coords,img_origin):
    scale_ver = 0.0953107555263 * 1e-3  # m/pix
    scale_hor = 0.074094097377  * 1e-3  # m/pix
    # Nico calculated these by hand
    # scale_hor = 0.00007396449
    # scale_ver = 0.00009842519

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

'''
    The z wire phantom has wires that all begin at x=0 and end at x=0.04 meters
    Viewed from above with the x axis down, all of the rightmost wires have a y value of 0.02
    The top two leftmost wires have a y value of 0. The bottom leftmost wire has a y value of 0.005
    The top central wire has a starting y value of 0.005 and ends with a y value of 0.015
    The center central wire has a starting y value of 0.015 and ends with a y value of 0.005
    The bottom central wire has a starting y value of 0.01 and an ending y value of 0.015
    There are three layers of wires, one at Z=0, another at Z=-0.01 and the last at Z=-0.02

    X=0.0  X-X---X  X=0.04 X-X---X
           X---X-X         X-X---X
           -X-X--X         -X--X-X   where - = 0.005
'''

wires = np.array([[[0.000, 0.000, 0.000],[0.040, 0.000, 0.000]],
                  [[0.000, 0.005, 0.000],[0.040, 0.015, 0.000]],
                  [[0.000, 0.020, 0.000],[0.040, 0.020, 0.000]],
                  [[0.000, 0.000,-0.010],[0.040, 0.000,-0.010]],
                  [[0.000, 0.015,-0.010],[0.040, 0.005,-0.010]],
                  [[0.000, 0.020,-0.010],[0.040, 0.020,-0.010]],
                  [[0.000, 0.005,-0.020],[0.040, 0.005,-0.020]],
                  [[0.000, 0.010,-0.020],[0.040, 0.015,-0.020]],
                  [[0.000, 0.020,-0.020],[0.040, 0.020,-0.020]]])

def translate(value, leftMin, leftMax, rightMin, rightMax):
    # Figure out how 'wide' each range is
    leftSpan = leftMax - leftMin
    rightSpan = rightMax - rightMin

    # Convert the left range into a 0-1 range (float)
    valueScaled = float(value - leftMin) / float(leftSpan)

    # Convert the 0-1 range into a value in the right range.
    return rightMin + (valueScaled * rightSpan)

def img_3d_coords_zFrame (image_coords_cent ): 
    ic = image_coords_cent
    assert(len(ic) == 9)
    z_wire_coords = np.zeros((9, 3))

    # Calculate center points
    for idx in [1, 4, 7]:
        center = idx
        left   = idx - 1
        right  = idx + 1
        ratioIm = np.linalg.norm(ic[center] - ic[left]) / np.linalg.norm(ic[right] - ic[left])
        # Set ratio zero correctly
        start = abs(wires[center,0,1] - wires[left,0,1]) / (wires[right,0,1] - wires[left,0,1])
        end   = abs(wires[center,1,1] - wires[left,0,1]) / (wires[right,0,1] - wires[left,0,1])
        ratio = translate(ratioIm, start, end, 0,1)
        if ratio < 0 or ratio > 1:
            return None
        z_wire_coords[idx] = wires[center][0] * (1-ratio) + wires[center][1] * ratio

    # Calculate normal of plane formed by center points
    a = z_wire_coords[7] - z_wire_coords[1]
    b = z_wire_coords[4] - z_wire_coords[1]
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    normal_guess = np.cross(a, b)
    if np.linalg.norm(normal_guess) < 0.1:
        return None # Central points are nearly colinear. Return nothing
    elif normal_guess[0] < 0:
        normal_guess = normal_guess * -1
    # Calculate left and right points
    for  idx in [1, 4, 7]:
        center = idx
        left   = idx - 1
        right  = idx + 1
        ldist = np.linalg.norm(ic[center]-ic[left])
        rdist = np.linalg.norm(ic[center]-ic[right])
        z_wire_coords[left]  = wires[left, 0]
        z_wire_ldistsq = ldist**2 - (wires[left, 0,1] - z_wire_coords[idx][1])**2
        z_wire_rdistsq = rdist**2 - (wires[right,0,1] - z_wire_coords[idx][1])**2
        if z_wire_ldistsq < 0 or z_wire_rdistsq < 0: # Violates triange inequality
            return None
        z_wire_coords[right] = wires[right,0]
        if normal_guess[1] > 0: 
            z_wire_coords[left][0]  = z_wire_coords[idx][0] + np.sqrt(z_wire_ldistsq)
            z_wire_coords[right][0] = z_wire_coords[idx][0] - np.sqrt(z_wire_rdistsq)
        else:
            z_wire_coords[left][0]  = z_wire_coords[idx][0] - np.sqrt(z_wire_ldistsq)
            z_wire_coords[right][0] = z_wire_coords[idx][0] + np.sqrt(z_wire_rdistsq)

    return z_wire_coords

def angle_between_vec(v1,v2):
    v1_u = v1/np.linalg.norm(v1)
    v2_u = v2/np.linalg.norm(v2)
    theta = np.arccos(v1_u.dot(v2_u))
    return theta


def find_img_frame_position(image_coords_cent,P3D): 
    p1, p2, p3, p4, p5, p6, p7, p8, p9 = image_coords_cent
    P1, P2, P3, P4, P5, P6 = P3D[0:6]

    theta =  angle_between_vec(p1, p2-p1)
    l_p1 = np.linalg.norm(p1)

    img_frame_position = P1 - l_p1*np.cos(theta)* (P2-P1)/np.linalg.norm(P2-P1) \
                        - l_p1*np.sin(theta)* (P4-P1)/np.linalg.norm(P4-P1) 

    return img_frame_position

def refine_R(R):
    U,S,Vt = np.linalg.svd(R)
    # R = U.dot(Vt.T)
    return  U.dot(Vt)

def find_img_frame_axis(image_coords_cent,P3D): 
    p1, p2, p3, p4, p5, p6, p7, p8, p9 = image_coords_cent

    P1, P2, P3, P4, P5, P6 = P3D[0:6]

    theta =  -angle_between_vec( p3-p1 , np.array([1,0]))
    x_axis = np.cos(theta)* (P3-P1)/np.linalg.norm(P3-P1) \
                + np.sin(theta) * (P1-P4)/np.linalg.norm(P1-P4)  ## explain 

    P3_cent = P3D - np.mean(P3D, axis=0)
    U,S,Vt =  np.linalg.svd(P3_cent)
    V = Vt.T
    z_axis = V[:,-1]
    z_dir_vec = np.cross(P3-P1,P4-P1 )


    if z_axis.dot(z_dir_vec) < 0:
        z_axis = -1*z_axis

    y_axis = np.cross(z_axis,x_axis) ## calculate with geometry

    R = np.zeros((3,3))
    R[:,0] = x_axis
    R[:,1] = y_axis
    R[:,2] = z_axis

    R_refined = refine_R(R)
    return R_refined


# def find_transf(image_coords,img_origin):
#   image_coords_cent = centralize_and_scale(image_coords,img_origin)
#   P3D = img_3d_coords_zFrame (image_coords_cent)
#   img_frame_orig = find_img_frame_orig(image_coords_cent,P3D)
#   R = find_img_frame_axis(image_coords,P3D)

#   return R,img_frame_orig

'''
image ids for transf2 80 102 125 177 255 321 363 390 490 664 738 800
'''
if __name__ == '__main__':
    # args = sys.argv
    # if len(args) < 3:
    #   print( "\n Usage: python frame_from_coords_clean.py <rosbag_data_path> <image_number>" )
    #   sys.exit()
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
    np.random.seed(6)
    for image_id in image_numbers:
        print(image_id,"image_id")
 
        pix_coords_path = save_dir + "/image_frame_coords/pix_coords_img_{}.txt".format(image_id)
        image_coords = np.loadtxt(pix_coords_path)
        
        # R,img_frame_orig = find_transf(image_coords,img_origin)
        image_coords_cent = centralize_and_scale(image_coords,img_origin)


        P3D = img_3d_coords_zFrame (image_coords_cent)
        if P3D is None:
            print("Image %s has colinear points and is unsuitable for calibration" % image_id)
            continue

        t = find_img_frame_position(image_coords_cent,P3D)
        R = find_img_frame_axis(image_coords,P3D)
        eul_out =  t3d.euler.mat2euler(R)
        print(np.array(eul_out)*180/np.pi,"eul_out deg")
        ultrasound_frames_list.append( create_transf_mat(R,t))
        x_axis,y_axis,z_axis = R[:,0], R[:,1], R[:,2]   

        c = np.random.rand(3,)
        for i in range(0, len(P3D)):
            ax3D.scatter(P3D[i,0], P3D[i,1], P3D[i,2], s=10,  marker='o', c=c)
            ax3D.plot(wires[i,:,0], wires[i,:,1], wires[i,:,2], color=(0.8,0.8,1.0))
            ax3D.text(P3D[i,0], P3D[i,1], P3D[i,2], 'P{}'.format(i))    

        ax3D.quiver(t[0], t[1], t[2], \
                    0.01 * x_axis[0], \
                    0.01 * x_axis[1], \
                    0.01 * x_axis[2]) # plotting x_axis  
        ax3D.quiver(t[0], t[1], t[2], \
                    0.01 * y_axis[0], \
                    0.01 * y_axis[1], \
                    0.01 * y_axis[2]) # plotting x_axis  
        ax3D.quiver(t[0], t[1], t[2], \
                    0.01 * z_axis[0], \
                    0.01 * z_axis[1], \
                    0.01 * z_axis[2]) # plotting x_axis  


        # ax3D.scatter(t[0], t[1], t[2], s=100,  marker='o', c=c)  
        ax3D.text(t[0], t[1], t[2], ' {}'.format(image_id))      
        ax3D.set_xlabel('x_axis')
        ax3D.set_ylabel('y_axis')
        ax3D.set_zlabel('z_axis')

    ultrasound_frames_file = save_dir + "/extra_files/ultrasound_frames.npz"
    np.set_printoptions(precision=3)
    # print(np.array(ultrasound_frames_list),"ultrasound_frames_list")
    np.savez(ultrasound_frames_file, ultrasound_frames=np.array(ultrasound_frames_list))

    ax3D.set_aspect('equal')
    plt.show()

        