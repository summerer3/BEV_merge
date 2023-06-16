import glob,cv2,os
import numpy as np
from tools.undist import undist
import matplotlib.pyplot as plt

def get_t(cam,PROJECT):
    t = (PROJECT[:,:,1]>0)
    if cam in ['front','back']:
        sum_t=np.sum(t,axis=1).tolist()
        mid = (np.min(sum_t.index(0))+np.max(sum_t.index(0)))/2
        if cam == 'front':
            t[np.min(sum_t.index(0))-50:,:]=0
        elif cam == 'back':
            t[:np.max(sum_t.index(0))+110,:]=0
    elif cam in ['left','right']:
        sum_t=np.sum(t,axis=0).tolist()
        mid = (np.min(sum_t.index(0))+np.max(sum_t.index(0)))/2
        if cam == 'left':
            t[:,np.min(sum_t.index(0))-60:]=0
        elif cam == 'right':
            t[:,:np.max(sum_t.index(0))+125]=0
            
    return t

def luminance_balance(images):
    [front,back,left,right] = [cv2.cvtColor(image,cv2.COLOR_BGR2HSV) 
                               for image in images]
    hf, sf, vf = cv2.split(front)
    hb, sb, vb = cv2.split(back)
    hl, sl, vl = cv2.split(left)
    hr, sr, vr = cv2.split(right)
    V_f = np.mean(vf)
    V_b = np.mean(vb)
    V_l = np.mean(vl)
    V_r = np.mean(vr)
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

def thin_mask(mask,ts,ang):
    h,w=np.shape(mask)
    ang=ang/180*np.pi
    
    offset=300
    # left up
    mask0=np.zeros([h,w])
    mask0[:int(h/2),:int(w/2)]=1
    mask1=mask*mask0
    idx=np.where(mask1==1)
    idx=np.asarray(idx,dtype='int')
    y_lu=np.max(idx[0,:])
    x_lu=np.max(idx[1,:])
    k0=(y_lu-0)/(x_lu-offset)
    k_lu_up=np.tan(np.arctan(k0)+ang)
    k_lu_dowm=np.tan(np.arctan(k0)-ang)
    X,Y=np.meshgrid(np.linspace(0,h-1,h),np.linspace(0,w-1,w))
    t_one=1*np.multiply((k_lu_up*(X-x_lu)+y_lu)<Y, (k_lu_dowm*(X-x_lu)+y_lu)>Y)
    mask_lu=mask1*t_one
    
    # plt.imshow(mask_lu)
    # plt.show()
    # right up
    mask0=np.zeros([h,w])
    mask0[:int(h/2),int(w/2):]=1
    mask1=mask*mask0
    idx=np.where(mask1==1)
    idx=np.asarray(idx,dtype='int')
    y_ru=np.max(idx[0,:])
    x_ru=np.min(idx[1,:])
    k0=(y_ru-0)/(x_ru-w+offset)
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
    k0=(y_ld-h)/(x_ld-offset)
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
    k0=(y_rd-h)/(x_rd-w+offset)
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
    print(ts[1].shape)
    
    _,h,w,_=np.shape(images)
    mask=np.sum(ts,axis=0)
    mask2=(mask==2)
    
    mask2,ts=thin_mask(mask2,ts,10)
    
    mh=int(h/2)
    mw=int(w/2)
    
    frt=np.asarray(ts[0], dtype='float')
    bck=np.asarray(ts[1], dtype='float')
    lft=np.asarray(ts[2], dtype='float')
    rgt=np.asarray(ts[3], dtype='float')
    
    p1=1
    p2=1
    # left front 
    for y in range(mh):
        if any(mask2[y,:mw]):
            ms = mask2[y,:mw]
            idxs=np.where(ms==True)
            lens=np.sum(ms!=0)
            weight = (idxs - np.min(idxs))/lens
            frt[y,idxs] = weight**p1
            lft[y,idxs] = 1-weight**p1
            
            # frt[y,idxs] = 0.5
            # lft[y,idxs] = 0.5
        else:
            break
    # right front 
    for y in range(mh):
        if any(mask2[y,mw:]):
            ms = mask2[y,mw:]
            idxs=np.asarray(np.where(ms==True), dtype='int')+mw
            lens=np.sum(ms!=0)
            weight = 1 - (idxs - np.min(idxs))/lens
            frt[y,idxs] = weight**p1
            rgt[y,idxs] = 1-weight**p1
            
            # frt[y,idxs] = 0.5
            # rgt[y,idxs] = 0.5
        else:
            break
    # left bottom
    for y in range(h-1,mh,-1):
        if any(mask2[y,:mw]):
            ms = mask2[y,:mw]
            idxs=np.where(ms==True)
            lens=np.sum(ms!=0)
            weight = (idxs - np.min(idxs))/lens
            bck[y,idxs] = weight**p1
            lft[y,idxs] = 1-weight**p1
            
            # bck[y,idxs] = 0.5
            # lft[y,idxs] = 0.5
        else:
            break
    # right bottom 
    for y in range(h-1,mh,-1):
        if any(mask2[y,mw:]):
            ms = mask2[y,mw:]
            idxs=np.asarray(np.where(ms==True), dtype='int')+mw
            lens=np.sum(ms!=0)
            weight = 1 - (idxs - np.min(idxs))/lens
            bck[y,idxs] = weight**p2
            rgt[y,idxs] = 1-weight**p2
            
            # bck[y,idxs] = 0.5
            # rgt[y,idxs] = 0.5
        else:
            break
    
    ts=[frt,bck,lft,rgt]
    
    return ts
    
def merge_BEV(cam_list):
    UAV_path = glob.glob('data/BEV_data/UAV_data/*.*')[0]
    UAV=cv2.imread(UAV_path)
    h,w,_=np.shape(UAV)
    
    front_pth=glob.glob('zhixng/front/*.jpg')
    back_pth=glob.glob('zhixng/back/*.jpg')
    left_pth=glob.glob('zhixng/left/*.jpg')
    right_pth=glob.glob('zhixng/right/*.jpg')
    front_pth.sort(key=lambda x:int(x.split('_')[-1].split('.')[-2]))
    back_pth.sort(key=lambda x:int(x.split('_')[-1].split('.')[-2]))
    left_pth.sort(key=lambda x:int(x.split('_')[-1].split('.')[-2]))
    right_pth.sort(key=lambda x:int(x.split('_')[-1].split('.')[-2]))
    
    num_png=len(right_pth)
    
    mask_path='data/mask.npy'
    mask_exist=os.path.exists(mask_path)
    
    if mask_exist==1:
        print('mask exist!')
        ts0=np.load(mask_path)
    
    size=(w,h)
    # save video
    # fourcc=cv2.VideoWriter_fourcc(*'XVID')
    # fps=5
    # out=cv2.VideoWriter('data/out.avi',fourcc,fps,size)
    
    for num in range(1):
        # os.system('clear')
        BEV_merge=0
        images=[]
        ts=[]
        for cam in cam_list:
            img = cv2.imread(glob.glob('data/test_data/'+cam+'/*.jpg')[-1])
            # img_nm=eval(cam+'_pth[num]')
            # img = cv2.imread(img_nm)
            
            
            img2 = undist(cam,img)
            H = np.load('data/intrinsic/'+cam+'/camera_'+cam+'_H.npy')
            print(H)
            PROJECT = cv2.warpPerspective(img2,H,(w,h))
            
            
            if mask_exist==0:
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
        for img in images:
            for jj in range(3):
                img[:,:,jj]=img[:,:,jj]*ts[ii]
            
            images[ii]=img
            ii+=1
        
        # balance HSV
        images = luminance_balance(images)
        
        # images = white_balance(images)
        images = np.asarray(images,dtype='float')
        # merge images
        for img in images:
            BEV_merge+=img/255*1.2
        
        weight_GT=0
        a,b=np.load('data/carcenter.npy')
        H=np.float32([[1,0,-a+w/2],[0,1,-b+h/2]])
        UAV=cv2.warpAffine(UAV,H,(w,h))
        BEV_merge=BEV_merge*(1-weight_GT)+UAV*weight_GT/255
        
        
        cv2.namedWindow('surround',1)
        cv2.imshow('surround', BEV_merge)
        cv2.waitKey(0)
        
        cv2.imwrite('result.jpg', BEV_merge*255)
    #     BEV_merge=np.asarray(BEV_merge*225,dtype='uint8')
    #     BEV_merge[BEV_merge>225]=225
    #     out.write(BEV_merge)
    
    # out.release()