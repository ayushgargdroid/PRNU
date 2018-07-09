import os
import cv2
import numpy as np
import requests
import shutil

file = open('db.txt')
for i in file.readlines():
    link = i[:-1]
    print(link)
    folder_name = link[link.rfind('/')+1:]
    file_name = folder_name[folder_name.rfind('_')+1:]
    folder_name = folder_name[:folder_name.rfind('_')]
    curr_folders = os.listdir(os.curdir+'/data/')
    if folder_name not in curr_folders:
        os.mkdir('data/'+folder_name)
    try:
        r = requests.get(link,stream=True,timeout=5)
        if r.status_code == 200:
            with open('data/'+folder_name+'/'+file_name,'wb') as t:
                r.raw.decode_content = True
                shutil.copyfileobj(r.raw, t)
    except Exception as e:
        print('Exception '+str(e))