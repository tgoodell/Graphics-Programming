import numpy as np
import cv2
import time
import random

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# ~ print(eye_cascade)
# ~ input()

cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
cv2.namedWindow("frame", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("frame",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
ESC=27

img=cv2.imread("pumpkin.png")
# ~ y,x=np.float32(np.mgrid[:h,:w])

x,y=0,0
eye_spread=140
width=640
height=360
eye_height=110
amp=50
while 1:
    ret, frame = cap.read()
    # ~ frame=(frame-127.0)*1.9+127
    # ~ frame[frame<0]=0
    # ~ frame[frame>255]=255
    # ~ frame=np.uint8(frame)
    #int flags=0, Size minSize=Size(), Size maxSize
    faces = face_cascade.detectMultiScale(frame[:,:,1],scaleFactor=1.1,minNeighbors = 5)
    if len(faces)>0:
        i,j,w,h=faces[0]
        x=.9*x+.1*(i+w//2)/640
        y=.9*y+.1*(j+h//2)/360
    else:
        x=.9*x+.1*random.random()
        y=.9*y+.1*random.random()
    
    cimg=img*1
    
    lx=width/2-eye_spread/2-amp*x+amp/2
    ly=ry=eye_height+amp*y-amp/2
    rx=width/2+eye_spread/2-amp*x+amp/2
    cv2.circle(cimg, (int(lx),int(ly)), 20, 0, -1)
    cv2.circle(cimg, (int(rx),int(ry)), 26, 0, -1)

    

    cv2.imshow('frame',cimg)
    key=cv2.waitKey(1)
    if key==ESC:
        break


# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
