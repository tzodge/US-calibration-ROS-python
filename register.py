
import numpy as np
import transforms3d as t3d

def register(P, Q):
	# Rigidly (+scale) aligns two point clouds with know point-to-point correspondences
	# with least-squares error.
	# Returns (scale factor c, rotation matrix R, translation vector t) such that
	#   Q = P*cR + t
	# if they align perfectly, or such that
	#   SUM over point i ( | P_i*cR + t - Q_i |^2 )
	# is minimised if they don't align perfectly.
    assert P.shape == Q.shape
    n, dim = P.shape

    centeredP = P - P.mean(axis=0)
    centeredQ = Q - Q.mean(axis=0)

    C = np.dot(np.transpose(centeredP), centeredQ) / n

    U, S, Vt = np.linalg.svd(C)
    d = (np.linalg.det(U) * np.linalg.det(Vt)) < 0.0

    if d:
        S[-1] = -S[-1]
        U[:, -1] = -U[:, -1]

    R = np.dot(U, Vt).T

    varP = np.var(P, axis=0).sum()
    # c = 1/varP * np.sum(S) # scale 
    c = 1 # scale 

    t = Q.mean(axis=0) - P.mean(axis=0).dot(c*R.T)

    return  R, t

if __name__ == '__main__':
    
     
    n=25

    pointcloud1 = np.random.rand(n,3) - np.array([0.5,0.5,0.5])

    axis = np.random.rand(3,) - np.array([0.5,0.5,0.5])
    axis = axis/np.linalg.norm(axis)
    rotation_angle = 2*(np.random.uniform()-0.5) * np.pi  

    rotation_matrix = t3d.axangles.axangle2mat(axis, rotation_angle)	 
    translation = np.random.rand(3,)
    pointcloud2 = rotation_matrix.dot(pointcloud1.T).T + translation

    R_pred, t_pred	 = register(pointcloud1,pointcloud2) 


    print(R_pred,"\npred rotation \n") 
    print(rotation_matrix,"gt rotation \n")

    print(t_pred,"pred translation\n") 
    print(translation,"gt translation \n") 