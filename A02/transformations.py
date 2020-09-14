import numpy as np
import cv2, math

def show(img,wait=0,destroy=True):
    img=np.uint8(img)
    cv2.imshow("image",img)
    cv2.waitKey(wait)
    if destroy:
        cv2.destroyAllWindows()

img=cv2.imread("input/squirrel.jif")
cube=cv2.imread("input/cube.jpg")
h,w=img.shape[:2]
angle=math.pi*30/180
T1=np.float64([[1,0,-w/2],[0,1,-h/2],[0,0,1]])
T2=np.float64([[1,0,w/2],[0,1,h/2],[0,0,1]])
R=np.float64([[np.cos(angle),-np.sin(angle),0],[np.sin(angle),np.cos(angle),0],[0,0,1]])
print(T1)
print(R)
print(T2)

M=R
print(M)

img2 = cv2.warpPerspective(img,M,(w*2,h*2))
# ~ img = cv2.warpPerspective(img,R,(w,h))
# ~ img = cv2.warpPerspective(img,T2,(w,h))
show(img2)

srcPoints=np.float32([[[0,0],[w,0],[w,h],[0,h]]])
# ~ print(M@np.float64([[0],[0],[1]]))
tp=cv2.perspectiveTransform(srcPoints,M)
print(tp)
maxX,maxY=tp.max(axis=1)[0]
minX,minY=tp.min(axis=1)[0]
print(minX,maxX,minY,maxY)
dstPoints=np.float32([[[50,32],[124,14],[179,41],[95,69]]])
M=cv2.getPerspectiveTransform(srcPoints, dstPoints)
cubeh,cubew=cube.shape[:2]
blank=img*0+255
squirrelPerspective=cv2.warpPerspective(img,M,(cubew,cubeh))
blankPerspective=cv2.warpPerspective(blank,M,(cubew,cubeh))

kernel=np.ones((5,5),np.uint8)
erosion=cv2.erode(blankPerspective,kernel,iterations=1)
show(blankPerspective)
# blur(blankPerspective)
blankPerspective=cv2.blur(blankPerspective,(5,5))
show(cube)

show(squirrelPerspective)
p=blankPerspective/255.0*0.9
cube=np.uint8(squirrelPerspective*p+cube*(1-p))
show(cube)
cv2.imwrite("output.png",cube)