import numpy as np
import cv2
import time

eyecacade=cv2.CascadeClassifier('haarcascade_eye.xml')
cap=cv2.VideoCapture(0)
cv2.namedWindow("frame",cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("frame",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
ESC=27

while 1:
    ret, frame=cap.read()
    eyes=eyecacade.detectMultiScale(frame[:,:,1],scaleFactor=1.2,minNeighbors=5)

    print(eyes)
    for (x,y,w,h) in eyes:
        #cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),5)
        darkness=1-frame[y:y+h,x:x+w,2]/255.0
        frame[y:y+h,x:x+w,0]=np.uint8(frame[y:y+h,x:x+w,0]*(1-darkness))
        frame[y:y + h, x:x + w, 1] = np.uint8(frame[y:y + h, x:x + w, 1] * (1 - darkness))
        frame[y:y + h, x:x + w, 2] = np.uint8(frame[y:y + h, x:x + w, 2] * (1 - darkness))+255*darkness


    cv2.imshow('frame',frame)
    key=cv2.waitKey(1)
    if key==ESC:
        break

cap.release()
cv2.destroyAllWindows()