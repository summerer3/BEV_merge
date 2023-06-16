import numpy as np
import cv2
import matplotlib.pyplot as plt

def undist(cam,img):
    K = np.load('data/intrinsic/'+cam+'/camera_'+cam+'_K.npy')
    D = np.load('data/intrinsic/'+cam+'/camera_'+cam+'_D.npy')
    print(K)
    print(D)

    h,w,_ = np.shape(img)

    FOCAL_SCALE=1
    SIZE_SCALE=4
    K_new = K.copy()

    K_new[0][0] *= FOCAL_SCALE
    K_new[1][1] *= FOCAL_SCALE
    K_new[0][2] = w / 2 * SIZE_SCALE
    K_new[1][2] = h / 2 * SIZE_SCALE
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3, 3), K_new, (int(w * SIZE_SCALE), int(h * SIZE_SCALE)), cv2.CV_16SC2)
    undist_img = cv2.remap(img, map1, map2, interpolation = cv2.INTER_LINEAR)
    
    return undist_img

def get_map(cam):
    K = np.load('../data/intrinsic/'+cam+'/camera_'+cam+'_K.npy')
    D = np.load('../data/intrinsic/'+cam+'/camera_'+cam+'_D.npy')
    img=cv2.imread('../data/BEV_data/fisheye_data/'+cam+'/'+cam+'.jpg')
    h,w,_=img.shape

    FOCAL_SCALE=1
    SIZE_SCALE=4
    K_new = K.copy()

    K_new[0][0] *= FOCAL_SCALE
    K_new[1][1] *= FOCAL_SCALE
    K_new[0][2] = w / 2 * SIZE_SCALE
    K_new[1][2] = h / 2 * SIZE_SCALE
    map1, _ = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3, 3), K_new, (int(w * SIZE_SCALE), int(h * SIZE_SCALE)), cv2.CV_16SC2)

    
    return map1[:,:,0],map1[:,:,1]