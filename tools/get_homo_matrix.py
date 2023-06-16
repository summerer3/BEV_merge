import cv2
import numpy as np

def get_HOMO_matrix(cam,fisheye,UAV):
    H,_ = cv2.findHomography(fisheye,UAV)
    print('Homo matrix of '+str(cam)+' is:\n')
    print(H)
    np.save('data/intrinsic/'+cam+'/camera_'+cam+'_H', H)