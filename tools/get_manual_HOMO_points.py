import cv2,glob,os
from tools.undist import undist
import numpy as np

def manual_HOMO_by_mouse(cam):
    r_resize=1
    
    def on_EVENT_RBUTTONDOWN(event, x, y,flags,param):
        if event == cv2.EVENT_RBUTTONDOWN:
            xy = "%d,%d" % (x, y)
            a.append(x)
            b.append(y)
            print(x,y)
            cv2.circle(img, (x, y), 1, (0, 0, 255), thickness=-1)
            cv2.putText(img, xy, (x, y), cv2.FONT_HERSHEY_PLAIN,
                        1.0, (0, 0, 0), thickness=1)
            cv2.imshow("image", img)
    
    if cam != 'UAV':
        path='data/BEV_data/fisheye_data/'+cam+'/*.jpg'
        path = glob.glob(path)
        img=cv2.imread(path[0])
        img=undist(cam, img)
    elif cam == 'UAV':
        path='data/BEV_data/UAV_data/*'
        path = glob.glob(path)
        img=cv2.imread(path[0])
        h,w,_=np.shape(img)
        if not os.path.exists('data/carcenter.npy'):
            a,b=[],[]
            cv2.namedWindow("image")
            cv2.setMouseCallback("image", on_EVENT_RBUTTONDOWN)
            cv2.imshow("image", img)
            cv2.waitKey(0)
            a=a[-1]
            b=b[-1]
            np.save('data/carcenter.npy',[a,b])
        else:
            a,b=np.load('data/carcenter.npy')
        # a,b=512,512
        H=np.float32([[1,0,w/2-a],[0,1,h/2-b]])
        img=cv2.warpAffine(img,H,(w,h))
    
    # get points (x,y) by right buttom
    print('=='*20)
    print('get HOMO points (x,y) by RIGHT buttom!')
    print('=='*20)

    img=cv2.resize(img,(int(np.shape(img)[1]/r_resize), 
                        int(np.shape(img)[0]/r_resize)))
    
    a,b=[],[]
    cv2.namedWindow("image",0)
    cv2.setMouseCallback("image", on_EVENT_RBUTTONDOWN)
    cv2.imshow("image", img)
    cv2.waitKey(0)
    
    homo_points=np.vstack([a,b])[:,-4:].T*r_resize
    # output the last 4 points
    return homo_points
