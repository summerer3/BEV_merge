import cv2,glob,os
from get_config import get_config
from undist import get_map
import numpy as np
import matplotlib.pyplot as plt

def get_t(cam,PROJECT):
    t = (PROJECT[:,:,1]>0)
    if cam in ['front','back']:
        sum_t=np.sum(t,axis=1).tolist()
        mid = (np.min(sum_t.index(0))+np.max(sum_t.index(0)))/2
        if cam == 'front':
            t[int(mid):,:]=0
        elif cam == 'back':
            t[:int(mid),:]=0
    elif cam in ['left','right']:
        sum_t=np.sum(t,axis=0).tolist()
        mid = (np.min(sum_t.index(0))+np.max(sum_t.index(0)))/2
        if cam == 'left':
            t[:,int(mid):]=0
        elif cam == 'right':
            t[:,:int(mid)]=0

    return t

def remap(img,map1,map2):
    newimg=np.zeros((map2.shape[0],map2.shape[1],3))
    for i in range(3):
        newimg[:,:,i]=img[map2,map1,i]/255.0
    newimg[newimg>1.1]=0
    
    return newimg

def get_mask(cam,map_pth):
    map=[]
    map1,map2 = get_map(cam)
    H = np.load('../data/intrinsic/'+cam+'/camera_'+cam+'_H.npy')
    map1 = cv2.warpPerspective(map1,H,(w,h))
    map2 = cv2.warpPerspective(map2,H,(w,h))
    map1[map1>fisheye.shape[1]-1]=fisheye.shape[1]-1
    map1[map1<0]=0
    
    map2[map2>fisheye.shape[0]-1]=fisheye.shape[0]-1
    map2[map2<0]=0
    
    map.append(map1)
    map.append(map2)
    np.save(map_pth,[map1,map2])
    
    return map1,map2

def thin_mask(mask,ts,ang):
    h,w=np.shape(mask)
    ang=ang/180*np.pi
    # left up
    mask0=np.zeros([h,w])
    mask0[:int(h/2),:int(w/2)]=1
    mask1=mask*mask0
    idx=np.where(mask1==1)
    idx=np.asarray(idx,dtype='int')
    y_lu=np.max(idx[0,:])
    x_lu=np.max(idx[1,:])
    k0=y_lu/x_lu
    k_lu_up=np.tan(np.arctan(k0)+ang)
    k_lu_dowm=np.tan(np.arctan(k0)-ang)
    X,Y=np.meshgrid(np.linspace(0,h-1,h),np.linspace(0,w-1,w))
    t_one=1*np.multiply((k_lu_up*(X-x_lu)+y_lu)<Y, (k_lu_dowm*(X-x_lu)+y_lu)>Y)
    mask_lu=mask1*t_one
    # right up
    mask0=np.zeros([h,w])
    mask0[:int(h/2),int(w/2):]=1
    mask1=mask*mask0
    idx=np.where(mask1==1)
    idx=np.asarray(idx,dtype='int')
    y_ru=np.max(idx[0,:])
    x_ru=np.min(idx[1,:])
    k0=(y_ru-0)/(x_ru-w)
    k_ru_up=np.tan(np.arctan(k0)+ang)
    k_ru_dowm=np.tan(np.arctan(k0)-ang)
    X,Y=np.meshgrid(np.linspace(0,h-1,h),np.linspace(0,w-1,w))
    t_one=1*np.multiply((k_ru_up*(X-x_ru)+y_ru)>Y, (k_ru_dowm*(X-x_ru)+y_ru)<Y)
    mask_ru=mask1*t_one
    # left down
    mask0=np.zeros([h,w])
    mask0[int(h/2):,:int(w/2)]=1
    mask1=mask*mask0
    idx=np.where(mask1==1)
    idx=np.asarray(idx,dtype='int')
    y_ld=np.min(idx[0,:])
    x_ld=np.max(idx[1,:])
    k0=(y_ld-h)/(x_ld-0)
    k_ld_up=np.tan(np.arctan(k0)+ang)
    k_ld_dowm=np.tan(np.arctan(k0)-ang)
    X,Y=np.meshgrid(np.linspace(0,h-1,h),np.linspace(0,w-1,w))
    t_one=1*np.multiply((k_ld_up*(X-x_ld)+y_ld)<Y, (k_ld_dowm*(X-x_ld)+y_ld)>Y)
    mask_ld=mask1*t_one
    # right down
    mask0=np.zeros([h,w])
    mask0[int(h/2):,int(w/2):]=1
    mask1=mask*mask0
    idx=np.where(mask1==1)
    idx=np.asarray(idx,dtype='int')
    y_rd=np.min(idx[0,:])
    x_rd=np.min(idx[1,:])
    k0=(y_rd-h)/(x_rd-w)
    k_rd_up=np.tan(np.arctan(k0)+ang)
    k_rd_dowm=np.tan(np.arctan(k0)-ang)
    X,Y=np.meshgrid(np.linspace(0,h-1,h),np.linspace(0,w-1,w))
    t_one=1*np.multiply((k_rd_up*(X-x_rd)+y_rd)>Y, (k_rd_dowm*(X-x_rd)+y_rd)<Y)
    mask_rd=mask1*t_one
    ## up
    ts[0]*=np.multiply((k_lu_dowm*(X-x_lu)+y_lu)>Y, (k_ru_up*(X-x_ru)+y_ru)>Y)
    ## down
    ts[1]*=np.multiply((k_ld_up*(X-x_ld)+y_ld)<Y, (k_rd_dowm*(X-x_rd)+y_rd)<Y)
    ## left 
    ts[2]*=np.multiply((k_lu_up*(X-x_lu)+y_lu)<Y, (k_ld_dowm*(X-x_ld)+y_ld)>Y)
    ## right
    ts[3]*=np.multiply((k_ru_dowm*(X-x_ru)+y_ru)<Y, (k_rd_up*(X-x_rd)+y_rd)>Y)
    
    mask_res=mask_lu+mask_ru+mask_ld+mask_rd
    
    return mask_res,ts

def soft_bounary(images,ts):
    _,h,w,_=np.shape(images)
    mask=np.sum(ts,axis=0)
    mask2=(mask==2)
    
    mask2,ts=thin_mask(mask2,ts,40)
    
    mh=int(h/2)
    mw=int(w/2)-100
    
    frt=np.asarray(ts[0], dtype='float')
    bck=np.asarray(ts[1], dtype='float')
    lft=np.asarray(ts[2], dtype='float')
    rgt=np.asarray(ts[3], dtype='float')
    
    
    # left front 
    for y in range(mh):
        if any(mask2[y,:mw]):
            ms = mask2[y,:mw]
            idxs=np.where(ms==True)
            lens=np.sum(ms!=0)
            weight = (idxs - np.min(idxs))/lens
            frt[y,idxs] = weight
            lft[y,idxs] = 1-weight
        else:
            break
    # right front 
    for y in range(mh):
        if any(mask2[y,mw:]):
            ms = mask2[y,mw:]
            idxs=np.asarray(np.where(ms==True), dtype='int')+mw
            lens=np.sum(ms!=0)
            weight = 1 - (idxs - np.min(idxs))/lens
            frt[y,idxs] = weight
            rgt[y,idxs] = 1-weight
        else:
            break

    # left bottom
    for y in range(h-1,mh,-1):
        if any(mask2[y,:mw]):
            ms = mask2[y,:mw]
            idxs=np.where(ms==True)
            lens=np.sum(ms!=0)
            weight = (idxs - np.min(idxs))/lens
            bck[y,idxs] = weight
            lft[y,idxs] = 1-weight
        else:
            break
    # right bottom 
    for y in range(h-1,mh,-1):
        if any(mask2[y,mw:]):
            ms = mask2[y,mw:]
            idxs=np.asarray(np.where(ms==True), dtype='int')+mw
            lens=np.sum(ms!=0)
            weight = 1 - (idxs - np.min(idxs))/lens
            bck[y,idxs] = weight
            rgt[y,idxs] = 1-weight
        else:
            break
    
    ts=[frt,bck,lft,rgt]
    
    return ts

def luminance_balance(images):
    [front,back,left,right] = [cv2.cvtColor(image,cv2.COLOR_BGR2HSV) 
                               for image in images]
    hf, sf, vf = cv2.split(front)
    hb, sb, vb = cv2.split(back)
    hl, sl, vl = cv2.split(left)
    hr, sr, vr = cv2.split(right)
    V_f = np.min(vf[vf>0])
    V_b = np.min(vb[vb>0])
    V_l = np.min(vl[vl>0])
    V_r = np.min(vr[vr>0])
    V_mean = (V_f + V_b + V_l +V_r) / 4
    vf = cv2.add(vf,(V_mean - V_f))
    vb = cv2.add(vb,(V_mean - V_b))
    vl = cv2.add(vl,(V_mean - V_l))
    vr = cv2.add(vr,(V_mean - V_r))
    r=11
    front = cv2.merge([hf,sf,vf])
    back = cv2.merge([hb,sb,vb])
    left = cv2.merge([hl,sl,vl])
    right = cv2.merge([hr,sr,vr])
    images = [front,back,left,right]
    images = [cv2.cvtColor(image,cv2.COLOR_HSV2BGR) for image in images]
    return images

if __name__=='__main__':
    config_pth='../config.json'
    cam_list,chess=get_config(config_pth)
    
    K,D,H=[],[],[]
    undist_map1,undist_map2=[],[]
    
    mask_path='../data/mask.npy'
    mask_exist=os.path.exists(mask_path)
    if mask_exist==1:
        print('mask exist!')
        ts0=np.load(mask_path)
    
    ts=[]
    images=[]
    BEV_merge=0
    mapy,mapx={},{}
    mask={}
    map_dir='../data/map'
    if os.path.exists(map_dir):
        pass
    else:
        os.mkdir(map_dir)
    for cam in cam_list:
        map_pth=map_dir+'/map_'+cam+'.npy'
        exist_map=os.path.exists(map_pth)
        
        UAV=cv2.imread(glob.glob('../data/BEV_data/UAV_data/*.*')[0])
        h,w,_=UAV.shape
        fisheye=cv2.imread(glob.glob('../data/BEV_data/fisheye_data/'+cam+'/*.jpg')[-1])
        if exist_map==0 or 1:
            
            map1,map2=get_mask(cam,map_pth)
            
            print('map SAVED!')
        else:
            print('map EXISTED!')
            map1,map2=np.load(map_pth)
        
        mapy[cam]=map1
        mapx[cam]=map2
        
        
        if mask_exist==0:
            PROJECT=remap(fisheye,map1,map2)
            t=get_t(cam,PROJECT)
            ts.append(t)
            for i in range(PROJECT.shape[2]):
                PROJECT[:,:,i]*=t
            images.append(PROJECT)
            
    if mask_exist==0:
        ts=soft_bounary(images,ts)
        np.save(mask_path,ts)
        print('MASK wroten!')
    else:
        ts=ts0
    ii=0
    for cam in cam_list:
        mask[cam]=ts[ii,:,:]
        ii+=1
    
    ## ======================================  cal  ==========================================
    path_frame=glob.glob('../src/*')
    # path_frame=glob.glob('../室内正常_晴天_多云_阴天_车位检测_垂直_T形_白色_水泥/博乐广场地下停车场/图片文件/*')
    
    path_frame=np.sort(path_frame)
    
    jj=0
    for i in range(len(path_frame)):
        print('='*100)
        front_pth=glob.glob(os.path.join(path_frame[i],'*_4.jpg'))
        back_pth=glob.glob(os.path.join(path_frame[i],'*_3.jpg'))
        left_pth=glob.glob(os.path.join(path_frame[i],'*_1.jpg'))
        right_pth=glob.glob(os.path.join(path_frame[i],'*_2.jpg'))
        
        # front_pth.sort(key=lambda x:int(x.split('_')[-1].split('.')[-2]))
        # back_pth.sort(key=lambda x:int(x.split('_')[-1].split('.')[-2]))
        # left_pth.sort(key=lambda x:int(x.split('_')[-1].split('.')[-2]))
        # right_pth.sort(key=lambda x:int(x.split('_')[-1].split('.')[-2]))
        
        num_png=len(right_pth)
        
        
        outdir='../result'
        if os.path.exists(outdir):
            pass
        else:
            os.mkdir(outdir)
        for num in range(num_png):
            BEV_merge=0
            for cam in cam_list:
                img_nm=eval(cam+'_pth[num]')
                fisheye = cv2.imread(img_nm)
                
                print(eval(cam+'_pth[num]').split('/')[-1])
                ##=============================================================
                project=remap(fisheye,mapy[cam],mapx[cam])
                # balance HSV
                # project = luminance_balance(project)
                for i in range(3):
                    project[:,:,i]*=mask[cam]
            
                BEV_merge+=project
            BEV_merge*=1.2
            cv2.imshow('1', BEV_merge)
            cv2.waitKey(1)
            jj+=1
            savenm=outdir+'/'+str(jj)+'.jpg'
            BEV_merge*=255.0
            cv2.imwrite(savenm, BEV_merge)