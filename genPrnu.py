import cv2
import numpy as np
import multiprocessing
import math
import rwt
from scipy import special
import csv
from random import *
import requests
import shutil
import sqlite3
import os
import imutils
import io

def adapt_array(arr):
    # SQL util function - convert np array to sql array
    out = io.BytesIO()
    np.save(out, arr)
    out.seek(0)
    return sqlite3.Binary(out.read())

def convert_array(text):
    # SQL util function - convert sql array to np array
    out = io.BytesIO(text)
    out.seek(0)
    return np.load(out)

def getNumber(s):
    no = ''
    i = -1
    while str.isdigit(s[i]):
        no += s[i]
        i -= 1
    no = no[::-1]
    return int(no)

def getKeys():
    # returns all user ids in the DB
    cur.execute('SELECT id FROM users')
    table_keys_t = cur.fetchall()
    table_keys = [i for i in range(len(table_keys_t))]
    for i in range(len(table_keys_t)):
        table_keys[i] = table_keys_t[i][0]
    del table_keys_t
    return table_keys

def mean2(x):
    y = np.sum(x) / np.size(x);
    return y

def corr2(a,b):
    a = a - mean2(a)
    b = b - mean2(b)

    r = (a*b).sum() / math.sqrt((a*a).sum() * (b*b).sum());
    return r

def threshold(y,t):
    res = y-t
    return (res+abs(res))/2

def removeNeighborhood(X,x,ssize):
    M,N = X.shape[0],X.shape[1]
    radius = int((ssize-1)/2)
    X = np.roll(X,radius-x[1]+1,axis=1)
    X = np.roll(X,radius-x[0]+1,axis=0)
    Y = X[ssize:,:ssize]
    Y = Y.flatten()
    Y = np.append(Y,X[M*ssize:].T)
    return Y

def qFunction(x):
    if(x<37.5):
        Q = 1/2*special.erfc(x/math.sqrt(2))
        logQ = math.log(Q)
    else:
        Q = (1/math.sqrt(2*math.pi))/np.multiply(x,np.exp(-np.divide(np.square(x),2)))
        logQ = -np.square(x)/2 - np.log(x) - 1/2*math.log(2*math.pi)
    return Q,logQ

def FAFromPCE(pce,search_space):
    p,logP = qFunction(np.sign(pce)*np.sqrt(np.abs(pce)))
    if(pce<50):
        FA = np.power(1-(1-p),search_space)
    else:
        FA = search_space * p
    if(FA==0):
        FA = search_space*p
        log10FA = np.log10(search_space) + logP*math.log10(2)
    else:
        log10FA = np.log10(FA)
    return FA,log10FA

def zeroMean(X):
    reshaped = 0
    if(len(X.shape)==2):
        X = np.reshape(X,[X.shape[0],X.shape[1],1])
        reshaped = 1
    M,N,K = X.shape[0],X.shape[1],X.shape[2]
    Y = np.zeros(X.shape)
    row = np.zeros([M,K])
    col = np.zeros([K,N])
    for i in range(K):
        t = np.mean(X[:,:,i])
        X[:,:,i] = X[:,:,i] - t
    for i in range(K):
        col[i,:] = np.mean(X[:,:,i],axis=0)
        row[:,i] = np.mean(X[:,:,i].T,axis=0).T
    for j in range(K):
        Y[:,:,j]=X[:,:,j]-np.ones([M,1])*col[j,:]
    for j in range(K):
        t = row[:,j]
        t.shape = [t.shape[0],1]
        Y[:,:,j]=Y[:,:,j]-t*np.ones([1,N])
    if reshaped is 1:
        Y = np.reshape(Y,[Y.shape[0],Y.shape[1]])
    return Y

def zeroMeanTotal(X):
    if len(X.shape) != 2:
        Y = np.zeros(X.shape)
        Z = zeroMean(X[::2,::2,:])
        Y[::2,::2,:] = Z
        Z = zeroMean(X[::2,1::2,:])
        Y[::2,1::2,:] = Z
        Z = zeroMean(X[1::2,::2,:])
        Y[1::2,::2,:] = Z
        Z = zeroMean(X[1::2,1::2,:])
        Y[1::2,1::2,:] = Z
    else:
        Y = np.zeros(X.shape)
        Z = zeroMean(X[::2,::2])
        Y[::2,::2] = Z
        Z = zeroMean(X[::2,1::2])
        Y[::2,1::2] = Z
        Z = zeroMean(X[1::2,::2])
        Y[1::2,::2] = Z
        Z = zeroMean(X[1::2,1::2])
        Y[1::2,1::2] = Z
    return Y

def waveNoise(coef,noiseVar):
    t = np.square(coef)
    filter = np.ones([3,3])/9
    coefVar = threshold(cv2.filter2D(t,-1,filter,borderType=cv2.BORDER_CONSTANT),noiseVar)
    for w in range(5,10,2):
        filter = np.ones([w,w])/(w*w)
        EstVar = threshold(cv2.filter2D(t,-1,filter,borderType=cv2.BORDER_CONSTANT),noiseVar)
        coefVar = np.minimum(coefVar, EstVar)
    return np.divide(np.multiply(coef,noiseVar),np.add(coefVar,noiseVar))

def noiseExtract(img,qmf,sigma,L):
    M,N = img.shape[0],img.shape[1]
    m = 2**L
    minpad = 2
    nr = math.ceil((M+minpad)/m)*m
    nc = math.ceil((N+minpad)/m)*m
    pr = math.ceil((nr-M)/2)
    prd= math.floor((nr-M)/2)
    pc = math.ceil((nc-N)/2)
    pcr= math.floor((nc-N)/2)
    t1 = np.insert(np.insert(img[pr-1::-1,pc-1::-1],pc,img[pr-1::-1,:].T,axis=1),N+pc,img[pr-1::-1,N-1:N-pcr-1:-1].T,axis=1)
    t2 = np.insert(np.insert(img[:,pc-1::-1],pc,img.T,axis=1),N+pc,img[:,N-1:N-pcr-1:-1].T,axis=1)
    t3 = np.insert(np.insert(img[M-1:M-prd-1:-1,pc-1::-1],pc,img[M-1:M-prd-1:-1,:].T,axis=1),N+pc,img[M-1:M-prd-1:-1,N-1:N-pcr-1:-1].T,axis=1)
    img = np.insert(np.insert(t2,[pr-1],t1,axis=0),pr-1+M,t3,axis=0)
    # img = np.float32([[img[pr:1:-1,pc:1:-1],img[pr:1:-1,:],img[pr:1:-1,N:N-pcr+1:-1]],[img[:,pc:1:-1],img,img[:,N:N-pcr+1:-1]],[img[M:M-prd+1:-1,pc:1:-1],img[M:M-prd+1:-1,:],img[M:M-prd+1:-1,N:N-pcr+1:-1]]])    
    img = np.float64(img)
    NoiseVar = sigma**2
    wave_trans = np.zeros([nr,nc])
    # print(img.shape)
    # cA,(cH,cV,cD) = pywt.dwt2(img,'db4','per')
    # wave_trans = np.append(np.append(cA,cH,0),np.append(cV,cD,0),1)
    wave_trans = rwt.dwt(img,qmf,L)[0]
    for i in range(L):
        # Hhigh = [k for k in range(int(nc/2)+1,nc+1)]
        # Hlow = [k for k in range(1,int(nc/2)+1)]
        # Vhigh = [k for k in range(int(nr/2)+1,nr+1)]
        # Vlow = [k for k in range(1,int(nr/2)+1)]
        wave_trans[0:int(nr/2),int(nc/2):nc] = np.around(waveNoise(wave_trans[0:int(nr/2),int(nc/2):nc],NoiseVar),4)
        wave_trans[int(nr/2):nr,0:int(nc/2)] =  np.around(waveNoise(wave_trans[int(nr/2):nr,0:int(nc/2)],NoiseVar),4)
        wave_trans[int(nr/2):nr,int(nc/2):nc] = np.around(waveNoise(wave_trans[int(nr/2):nr,int(nc/2):nc],NoiseVar),4)
        nc = int(nc/2)
        nr = int(nr/2)
    wave_trans[:nr,:nc] = 0
    # cA = wave_trans[:136,:136]
    # cV = wave_trans[:136,136:]
    # cH = wave_trans[136:,:136]
    # cD = wave_trans[136:,136:]
    # image_noise = pywt.idwt2((cA,(cH,cV,cD)),'db4','per')
    image_noise = np.around(rwt.idwt(wave_trans,qmf,L)[0],4)
    return image_noise[pr:pr+M,pc:pc+N]

def noiseExtractFromImg(img,sigma):
    L = 4
    qmf = np.array([0.2304,0.7148,0.6309,-0.0280,-0.1870,0.0308,0.0329,-0.0106],dtype=np.float64)
    noise = np.zeros(img.shape)
    for j in range(3):
        noise[:,:,j] = noiseExtract(img[:,:,j],qmf,sigma,L)
    noise = noise.astype(np.float32)
    noise = cv2.cvtColor(noise,cv2.COLOR_BGR2GRAY)
    noise = zeroMeanTotal(noise)
    return noise

def intenScale(c):
    T = 252
    v = 6
    out = np.exp(-1*np.divide(np.square(np.subtract(c,T)),v))
    out[c<T] = np.divide(c[c<T],T)
    return np.around(out,4)

def getSaturMap(X):
    X = X.astype(np.uint8)
    M,N = X.shape[0],X.shape[1]
    if(X.max()<=250):
        return np.ones(X.shape)
    Xh = X - np.roll(X,1,axis=1)
    Xv = X - np.roll(X,1,axis=0)
    saturMap = np.bitwise_and(Xh,np.bitwise_and(Xv,np.bitwise_and(np.roll(Xh,-1,axis=1),np.roll(Xv,-1,axis=0))))
    if(len(X.shape)==3):
        for i in range(3):
            maxI[i] = X[:,:,i].max()
            if maxI[i] > 250:
                saturMap[:,:,j] = np.bitwise_not(np.bitwise_and(np.uint8((X[:,:,i]==maxI[i])*1),np.bitwise_not(saturMap[:,:,i])))
    else:
        maxX = X.max()
        saturMap = np.bitwise_not(np.bitwise_and(np.uint8((X==maxX)*1),np.bitwise_not(saturMap)))

    return (saturMap>=255)*1

def wienerFilter(noise,sigma):
    F = np.fft.fft2(noise)
    Fmag = np.abs(np.divide(F,math.sqrt(noise.shape[0]*noise.shape[1])))
    noiseVar = sigma**2
    Fmag1 = waveNoise(Fmag,noiseVar)
    Fmag[np.where(Fmag==0)] = 1
    Fmag1[np.where(Fmag==0)] = 0
    F = np.multiply(F,np.divide(Fmag1,Fmag))
    return np.around(np.real(np.fft.ifft2(F)),4)

def getFingerprint(imgs):
    sigma = 3
    L = 4
    qmf = np.array([0.2304,0.7148,0.6309,-0.0280,-0.1870,0.0308,0.0329,-0.0106],dtype=np.float64)
    RPsum = np.zeros([imgs.shape[3],imgs.shape[1],imgs.shape[2]])
    RP = np.zeros([imgs.shape[3],imgs.shape[1],imgs.shape[2]])
    NN = np.zeros([imgs.shape[3],imgs.shape[1],imgs.shape[2]])
    for i in imgs:
        i = i.astype(np.float32,copy=False)
        for j in range(3):
            imNoise = noiseExtract(i[:,:,j],qmf,sigma,L)
            inten = np.multiply(intenScale(i[:,:,j]),getSaturMap(i[:,:,j]))
            RPsum[j] = RPsum[j] + np.multiply(imNoise,inten)
            NN[j] = NN[j] + np.square(inten)
    
    for j in range(3):
        RP[j] = np.divide(RPsum[j],NN[j]+1)
    RP = cv2.merge(RP)
    RP = zeroMeanTotal(RP)
    RP = RP.astype(np.float32)
    return np.around(cv2.cvtColor(RP,cv2.COLOR_RGB2GRAY),4)

def getCorr(X,Y):
    X = np.subtract(X,np.mean(X))
    Y = np.subtract(Y,np.mean(Y))
    tiltedY = np.fliplr(Y)
    tiltedY = np.flipud(tiltedY)
    TA = np.fft.fft2(tiltedY)
    FA = np.fft.fft2(X)
    FF = np.multiply(TA,FA)
    return np.real(np.fft.ifft2(FF))

def getPCE(C):
    squaresize = 11
    shift_range = np.uint8([0,0])
    if np.any(shift_range>=C.shape):
        shift_range = np.minimum(shift_range,C.shape-1)
    cInRange = C[C.shape[0]-shift_range[0]-1:,C.shape[1]-shift_range[1]-1:]
    (ypeak,xpeak) = np.unravel_index(np.argmax(cInRange, axis=None), cInRange.shape)
    peakHeight = cInRange[ypeak,xpeak]
    peakLocation = shift_range+np.uint8([1,1])-np.uint8([ypeak,xpeak])
    cWithoutPeak = removeNeighborhood(C,np.uint8([ypeak,xpeak]),squaresize)
    correl = C[C.shape[0]-1,C.shape[1]-1]
    pceEnergy = np.mean(np.multiply(cWithoutPeak,cWithoutPeak))
    PCE = peakHeight**2/pceEnergy*np.sign(peakHeight)
    pValue = 1/2*special.erfc(peakHeight/math.sqrt(pceEnergy)/math.sqrt(2))
    pFA,log10pFA = FAFromPCE(PCE,(shift_range[0]+1)*(shift_range[1]+1))
    return pValue,PCE

def getResults(fingerprint1,img):
    test_img = img
    noisex = noiseExtractFromImg(test_img,2)
    noisex = wienerFilter(noisex,np.std(noisex))
    fingerprint1 = fingerprint1.astype(np.float32)
    # fingerprint2 = fingerprint2.astype(np.float32)
    noisex = noisex.astype(np.float32)
    img = img.astype(np.float32,copy=False)
    Ix = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    C = getCorr(noisex,np.multiply(Ix,fingerprint1))
    # C = getCorr(fingerprint2,fingerprint1)
    x,y = getPCE(C)
    return 1.0-x,y

def getFingerprintUtil(imgs):
    RP = getFingerprint(imgs)
    sigmaRP = np.std(RP)
    return wienerFilter(RP,sigmaRP)
    # if j<=1:
    #     RP = getFingerprint(imgs[:1,...])
    #     sigmaRP = np.std(RP)
    #     return wienerFilter(RP,sigmaRP)
    # elif j<=3:
    #     RP = getFingerprint(imgs[:2,...])
    #     sigmaRP = np.std(RP)
    #     return wienerFilter(RP,sigmaRP)
    # else:
    #     RP = getFingerprint(imgs[:3,...])
    #     sigmaRP = np.std(RP)
    #     return wienerFilter(RP,sigmaRP)

def checkDB(idc,img,sensor_res):
    keys = getKeys()
    print(keys)
    for i in range(len(keys)):
        cur.execute('SELECT fingerprint,width,height,pce1,pce2 from users where id=?',(keys[i],))
        tt = cur.fetchone()
        fp = tt[0]
        width = tt[1]
        height = tt[2]
        pce_val1 = tt[3]
        pce_val2 = tt[4]
        # print(img.shape)
        # print(str(width)+' , '+str(height))
        if(sensor_res[0]!=width or sensor_res[1]!=height):
            continue
        fp = fp.astype(np.float32)
        pce = getKorus(fp,img[0])
        pce2 = getKorus(fp,img[1])
        print('Attacker: '+str(idc)+' Defender: '+str(keys[i])+' PCE1: '+str(pce)+' PCE2: '+str(pce2)+' Req. PCE1: '+str(pce_val1)+' PCE2: '+str(pce_val2))
        max_val = 0
        if pce_val2<0.002 and pce_val2>0.001 and pce<0.002 and pce>0.001 and pce2<0.002 and pce2>0.001:
            print('<<<Duplicate - '+str(idc)+' and '+str(keys[i])+'>>>')
            writer.writerows([[idc,keys[i]]])
            continue
        if pce_val1<0.002 or pce_val2<0.002 or pce<0.002 or pce2<0.002:
            continue
        # if pce_val1>pce_val2:
        #     max_val = pce_val1
        #     min_val = pce_val2
        # else:
        #     max_val = pce_val2
        #     min_val = pce_val1
        max_val = pce_val2
        max_val = round(max_val,4)
        a = str(max_val)
        ind = a.rfind('0')
        if ind<=3:
            t = '0.0000'
            t = float(t[:ind+1]+'1')
            if pce>max_val-t and pce<max_val+t and pce2>max_val-t and pce2<max_val+t:
                print('<<<Duplicate - '+str(idc)+' and '+str(keys[i])+'>>>')
                writer.writerows([[idc,keys[i]]])

def getNearestNeighbour(idc,keys):
    max = 0.0
    cur.execute('SELECT fingerprint from users where id=?',(idc,))
    fingerPrint1 = cur.fetchone()[0]
    key = '0'
    for i in range(len(keys)):
        if keys[i]==idc:
            continue
        cur.execute('SELECT fingerprint from users where id=?',(keys[i],))
        fingerPrint2 = cur.fetchone()[0]
        fingerPrint1 = fingerPrint1.astype(np.float32)
        fingerPrint2 = fingerPrint2.astype(np.float32)
        C = getCorr(fingerPrint1,fingerPrint2)
        x,_ = getPCE(C)
        if x>max:
            max = x
            key = keys[i]
    return key

def highestPCE(fingerprint,imgs,ino,j):
    pceValues = np.zeros(j)
    for k in range(j):
        if k in ino:
            continue
        xt,pceCorr = getResults(fingerPrint,imgs[k])
        pceValues[k] = pceCorr
        if xt<0.85:
            verd = 'Rejected'
        else:
            verd = 'Accepted'
        # print('Image '+str(k)+' = '+str(pceCorr))
        #writer.writerows([[prevID,docsList[k],yt,xt,verd]])
    t = np.argmax(pceValues)
    if(pceValues[t]<=1):
        return fingerprint,ino
    ino.append(t)
    cluster_imgs = imgs[ino]
    return getFingerprintUtil(cluster_imgs),ino

def getKorus(fp,img):
    img = np.uint8(img)
    img_noise = noiseExtractFromImg(img,3)
    fp = fp * cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    fp = fp - np.mean(fp)
    img_noise = img_noise - np.mean(img_noise)
    n1 = np.sqrt(np.sum(img_noise*img_noise))
    n2 = np.sqrt(np.sum(fp*fp))
    return np.sum(img_noise*fp)/(n1*n2)

def getKorusFromPrnu(fp1,fp2):
    fp = fp1.copy()
    tt = fp2.copy()
    fp = fp * fp2
    fp = fp - np.mean(fp)
    tt = tt - np.mean(tt)
    n1 = np.sqrt(np.sum(tt*tt))
    n2 = np.sqrt(np.sum(fp*fp))
    return np.sum(tt*fp)/(n1*n2)

def getInitH(imgs):
    n = imgs.shape[0]
    fingerPrints = np.zeros(imgs.shape[:3])
    corrMat = np.zeros([n,n])
    for i in range(n):
        fingerPrints[i] = getFingerprintUtil(np.float32([imgs[i]]))
    for i in range(n):
        for j in range(n):
            if i is j:
                corrMat[i][j] = 0.0
                continue
            corrMat[i][j] = corr2(fingerPrints[i],fingerPrints[j])
    return corrMat

def getProjection(arr,random_arr):
    #proj = np.matmul(arr,random_arr)
    proj = arr
    proj = (proj>=0)*1
    proj = np.uint8(proj.flatten())
    return proj


def generate_prnu(i):
    print(i)
    imgs = []
    img_list = os.listdir('./'+i)
    imgs = np.array([cv2.imread(i+'/'+img_list[0])])
    for j in img_list[1:]:
        t = cv2.imread(i+'/'+j)
        if t.shape[0]==imgs.shape[2]:
            t = imutils.rotate_bound(t,90)
        imgs = np.append(imgs,[t],axis=0)
    print(imgs.shape)
    height,width = imgs.shape[1],imgs.shape[2]
    imgs_center = np.array([imgs[0,int(height/2)-256:int(height/2)+256,int(width/2)-256:int(width/2)+256]])
    for j in range(imgs.shape[0]-1):
        imgs_center = np.append([imgs_center,imgs[j+1,int(height/2)-256:int(height/2)+256,int(width/2)-256:int(width/2)+256]],axis=0)
    fp = getFingerprintUtil(imgs_center)
    np.save(i+str(0),fp)
    imgs_center = np.array([imgs[0,int(height/2)+256:int(height/2)+256+512,int(width/2)-256:int(width/2)+256]])
    for j in range(imgs.shape[0]):
        imgs_center = np.append(imgs_center,[imgs[j,int(height/2)+256:int(height/2)+256+512,int(width/2)-256:int(width/2)+256]],axis=0)
    fp = getFingerprintUtil(imgs_center)
    np.save(i+str(1),fp)
    imgs_center = np.array([imgs[0,int(height/2)-256-512:int(height/2)-256,int(width/2)-256:int(width/2)+256]])
    for j in range(imgs.shape[0]):
        imgs_center = np.append(imgs_center,[imgs[j,int(height/2)-256-512:int(height/2)-256,int(width/2)-256:int(width/2)+256]],axis=0)
    fp = getFingerprintUtil(imgs_center)
    np.save(i+str(2),fp)
    imgs_center = np.array([imgs[0,int(height/2)-256:int(height/2)+256,int(width/2)+256:int(width/2)+256+512]])
    for j in range(imgs.shape[0]):
        imgs_center = np.append(imgs_center,[imgs[j,int(height/2)-256:int(height/2)+256,int(width/2)+256:int(width/2)+256+512]],axis=0)
    fp = getFingerprintUtil(imgs_center)
    np.save(i+str(3),fp)
    imgs_center = np.array([imgs[0,int(height/2)-256:int(height/2)+256,int(width/2)-256-512:int(width/2)-256]])
    for j in range(imgs.shape[0]):
        imgs_center = np.append(imgs_center,[imgs[j,int(height/2)-256:int(height/2)+256,int(width/2)-256-512:int(width/2)-256]],axis=0)
    fp = getFingerprintUtil(imgs_center)
    np.save(i+str(4),fp)

os.chdir('data/')
camera_list = os.listdir(os.curdir)
pool = multiprocessing.Pool(processes=4)
pool.map(generate_prnu,camera_list)