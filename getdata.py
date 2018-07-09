import os
import cv2
import numpy as np
import requests
import multiprocessing
import shutil

def download_fast(i):
    link = i[:-1]
    print(link)
    folder_name = link[link.rfind('/')+1:]
    file_name = folder_name[folder_name.rfind('_')+1:]
    folder_name = folder_name[:folder_name.rfind('_')]
    curr_folders = os.listdir(os.curdir+'/data/')
    if folder_name not in curr_folders:
        os.mkdir('data/'+folder_name)
    try:
        r = requests.get(link,stream=True,timeout=10)
        if r.status_code == 200:
            with open('data/'+folder_name+'/'+file_name,'wb') as t:
                r.raw.decode_content = True
                shutil.copyfileobj(r.raw, t)
    except Exception as e:
        print('Exception '+str(e))


file = open('db.txt')
links = file.readlines()
links = links[links.index('http://forensics.inf.tu-dresden.de/ddimgdb/images/gallery/Panasonic_DMC-FZ50_1_27619.JPG\n'):]
pool = multiprocessing.Pool(processes=4)
pool.map(download_fast,links)