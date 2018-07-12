import os
import cv2
import imutils
import numpy as np
import requests
import multiprocessing
import shutil

def saveData(i):
    print(i)
    img_list = np.array(os.listdir(os.curdir+'/'+i))
    rand = np.random.randint(low=0,high=len(img_list)-1,size=5)
    img_list = img_list[rand]
    imgs = np.array([cv2.imread(i+'/'+img_list[0])])
    for j in img_list[1:]:
        print(i+'/'+j)
        t = cv2.imread(i+'/'+j)
        if t.shape[0]==imgs.shape[2] or t.shape[0]==imgs.shape[1]:
            pass
        else:
            continue
        if t.shape[0]==imgs.shape[2]:
            t = imutils.rotate_bound(t,90)
        imgs = np.append(imgs,[t],axis=0)
    height,width = imgs.shape[1],imgs.shape[2]
    imgs_center = np.array([imgs[0,int(height/2)-128:int(height/2)+128,int(width/2)-128:int(width/2)+128]])
    for j in range(imgs.shape[0]-1):
        imgs_center = np.append(imgs_center,[imgs[j+1,int(height/2)-128:int(height/2)+128,int(width/2)-128:int(width/2)+128]],axis=0)
    for j in range(imgs_center.shape[0]):
        cv2.imwrite(i+str(j)+'.jpg',imgs_center[j])

os.chdir('./data/')
camera_list = os.listdir(os.curdir)
camera_list.remove('prnu')
pool = multiprocessing.Pool(processes=4)
pool.map(saveData,camera_list)
