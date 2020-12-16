import numpy as np
from numpy import random
import cv2
import time

eyecacade=cv2.CascadeClassifier('haarcascade_eye.xml')

def blackWhite(img, threshold):
    bw = 1*greyscale(img)
    bw[np.uint8(bw) < threshold] = 0
    bw[np.uint8(bw) > threshold] = 255
    return bw

def normalize(img):
    out=img*1.0
    out-=out.min()
    out/=out.max()
    out*=255.999
    return np.uint8(out)

def greyscale(img):
    b = img[:, :, 0]*0.1
    g = img[:, :, 1]*0.7
    r = img[:, :, 2]*0.2
    img=b+g+r
    return img

def desaturate(img):
    desat = 1*np.double(img[:, :, :])
    greyImg = greyscale(img)

    # Actual math behind desat
    desat[:, :, :] = (desat[:, :, :] *(1 - 0.5)) + (greyImg[:,:,None] * 0.5)
    # Overflow Check
    desat[desat > 255] = 255
    desat[desat < 0] = 0

    return desat

def star(img,r,center=(0,0),n=5,cycles=2,color=255,rotation=0):
    xc,yc=center
    h,w,*_=img.shape
    Y,X=np.mgrid[:h,:w]
    X-=xc
    Y-=yc
    d=np.hypot(X,Y)
    theta=np.arctan2(Y,X)+np.pi/2-rotation
    theta=theta%(np.pi*2/n)
    theta=np.minimum(theta,np.pi*2/n-theta)
    XP=d*np.cos(theta)
    YP=d*np.sin(theta)
    phi=np.arctan2(YP,XP-r)
    tipAngle=np.pi-2*np.pi*cycles/n
    img[phi>np.pi-tipAngle/2]=color


cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
cv2.namedWindow("frame", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("frame",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
ESC=27
last=time.time()

ret, frame = cap.read()
h,w,*_=frame.shape
y,x=np.float32(np.mgrid[:h,:w])

warp=False
desature=False
nonlinear=False
matrix=False
eyez=False
bw=False

j = k = i = 0

fps=30
t=0
dx=10
dy=10
while 1:
    ret, frame = cap.read()
    current=time.time()
    fps=.99*fps+.01/(current-last)
    print(fps)
    last=current

    key = cv2.waitKey(1)
    if key != -1:
        print(key)
    if key == ESC:
        break
    if key%32==ord('1')%32:
        desature=True
        warp=False
        nonlinear=False
        matrix=False
        bw = False
        eyez = False
    if key%32==ord('2')%32:
        warp=True
        desature=False
        nonlinear=False
        matrix=False
        bw = False
        eyez = False
    if key%32==ord('3')%32:
        warp=False
        desature=False
        nonlinear=True
        matrix=False
        bw = False
        eyez = False
    if key%32==ord('4')%32:
        warp=False
        desature=False
        nonlinear=False
        matrix=True
        bw = False
        eyez=False
    if key%32==ord('5')%32:
        warp=False
        desature=False
        nonlinear=False
        matrix=False
        bw=False
        eyez=True
    if key%32==ord('6')%32:
        warp=False
        desature=False
        nonlinear=False
        matrix=False
        eyez=False
        bw=True

    if desature==True:
        frame=normalize(desaturate(frame))

    if warp==True:
        angle = np.pi * t / 1000 + np.pi / 3 * np.cos(t * .05)*2
        T1 = np.float64([[1, 0, w/3*np.cos(angle)],
                         [0, 1, h/3*np.cos(angle)],
                         [0, 0, 1]])
        frame = cv2.warpPerspective(frame, T1, (w, h))

    if nonlinear==True:
        frame=cv2.remap(frame,(x+dx*t)%(w-1),(y+dy*np.sin(np.pi*t/180))%(h-1),cv2.INTER_CUBIC)
        frame=normalize(greyscale(frame))

    if matrix==True:
        M = np.float32([[1, 0, np.sin(2*t+10)],
                        [0, 1, np.sin(2*t+10)],
                        [0, 0, 1]])
        frame = cv2.warpPerspective(frame, M, (w,h))
        frame = cv2.remap(frame, x + dx * np.sin(np.pi * (y + t)/180),y + dy * np.sin(np.pi * (x + t)/180), 1)

    if eyez==True:
        ret, frame = cap.read()
        eyes = eyecacade.detectMultiScale(frame[:, :, 1], scaleFactor=1.2, minNeighbors=5)

        yellow = (0, 255, 255)
        for (x, y, w, h) in eyes:
            star(frame,w//2,(x+w//2,y+h//2),color=yellow)

        c=0
        while c<3:
            star(frame, random.randint(1,5), (random.randint(0,640), random.randint(0,360)), color=yellow)
            c+=1

    if bw==True:
        frame=blackWhite(frame,110)


    if key % 32 == ord('s') % 32:
        cv2.imwrite("output/out.png", frame)
    t+=1

    cv2.imshow('frame',frame)

    dx=max(0,dx)
    dy=max(0,dy)

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
