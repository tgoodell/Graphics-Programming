import numpy as np
import cv2
import time

cap=cv2.VideoCapture(0)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

ESC=27
last=time.time()
h=360
w=480

cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
y,x=np.float32(np.mgrid[:h,:w])
fps=30
t=0

while 1:
    ret, frame=cap.read()
    current=time.time()
    fps=.99*fps+.01/(current-last)
    print(fps)
    last=current

    frame=cv2.remap(frame,x+10*np.sin(2*np.pi*(y+3*t)/100),y+10*np.sin(2*np.pi*(x+3*t)/100),0)
    t+=10

    #if key%32==ord('a')%32:
    #    dx+=1
    #if key % 32 == ord('a') % 32:
    #    dx-=1

    cv2.imshow('frame',frame)
    key=cv2.waitKey(1)
    if key==ESC:
        break

cap.release()
cv2.destroyAllWindows()