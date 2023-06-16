from tools.get_config import get_config
from tools.intrinsicCalib import main
from tools.get_manual_HOMO_points import manual_HOMO_by_mouse
from tools.get_homo_matrix import get_HOMO_matrix
from tools.project_BEV import merge_BEV
import os

if __name__ == '__main__':
    if os.path.exists('data/mask.npy'):
        os.remove('data/mask.npy')
    
    conf = 'config.json'
    # get camera list
    cam_list,chess = get_config(conf)
    # get homo matrix
    for cam in cam_list:
        t_K = os.path.exists('data/intrinsic/'+cam+'/camera_'+cam+'_K.npy')
        t_D = os.path.exists('data/intrinsic/'+cam+'/camera_'+cam+'_D.npy')
        t_H = os.path.exists('data/intrinsic/'+cam+'/camera_'+cam+'_H.npy')
        if not t_K*t_D or 0 :
            main(cam)
        if not t_H or 1:
            cam_homo_points = manual_HOMO_by_mouse(cam)
            UAV_homo_points = manual_HOMO_by_mouse('UAV')
            get_HOMO_matrix(cam,cam_homo_points,UAV_homo_points)
    # show the BEV merge image
    merge_BEV(cam_list)