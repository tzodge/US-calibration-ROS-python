import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

# fig = plt.figure()

# ax_handle = fig.add_subplot(111, projection='3d')

# P_h_array = np.random.rand(10,3)
# Q_h_array = P_h_array + 2


# ax_handle.scatter(P_h_array[:,0], P_h_array[:,1], P_h_array[:,2],  label='P')
# ax_handle.scatter(Q_h_array[:,0], Q_h_array[:,1], Q_h_array[:,2],  label='Q')

# for i in range(len(P_h_array)):
#     ax_handle.plot([P_h_array[i,0], Q_h_array[i,0]],\
#                    [P_h_array[i,1], Q_h_array[i,1]],\
#                    [P_h_array[i,2] ,Q_h_array[i,2]] )

# ax_handle.plot([np.arange(5),np.arange(5)+1],
#                 [np.arange(5),np.arange(5)+1],
#                 [np.arange(5),np.arange(5)+1])
# plt.show()

import numpy as np 
def transf_pixel_to_EE():

    '''
    output: Mat_pix_to_EE in R^{4x4}
    Note that,  Mat_pix_to_EE.[p_x,p+y,0,1] should give a 
    homogeneous coordinate in the robot EE fram
    '''
    img_origin = np.array([388,134])
    scale_ver = 0.0953107555263 * 1e-3  # m/pix
    scale_y = 0.0953107555263 * 1e-3  # m/pix
    scale_hor = 0.074094097377  * 1e-3  # m/pix 
    scale_x = 0.074094097377  * 1e-3  # m/pix 
    y_off, x_off = img_origin # pixels 

    Mat_scale = np.array(([[scale_x,    0,  0, -scale_x*x_off ],
                           [0,    scale_y,  0, -scale_y*y_off ],
                           [0,          0,  0,        0       ],
                           [0,          0,  0,        1       ]])) 

    Mat_hom = np.array([[ 0.96416, -0.1493 , -0.21932,  -0.2    ],
                        [-0.25585, -0.30439, -0.91754,  0.0     ],
                        [ 0.07023,  0.94077, -0.33168,  0.22    ],
                        [ 0.     ,  0.     ,  0.     ,  1.     ]])

    Mat_pix_to_EE = Mat_hom.dot(Mat_scale)

    return Mat_scale, Mat_hom, Mat_pix_to_EE


Mat_scale, Mat_hom, Mat_pix_to_EE = transf_pixel_to_EE()

print("Mat_scale: ", Mat_scale)
print("Mat_hom: ", Mat_hom)
print("Mat_pix_to_EE: ", Mat_pix_to_EE)
