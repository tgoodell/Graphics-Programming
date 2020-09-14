import numpy as np
import cv2, math

def show(img,wait=0,destroy=True):
    img=np.uint8(img)
    cv2.imshow("image",img)
    cv2.waitKey(wait)
    if destroy:
        cv2.destroyAllWindows()

img1=cv2.imread("input/image1.png")
img2=cv2.imread("input/image2.png")
cube=cv2.imread("input/cube.png")
h1,w1=img1.shape[:2]
h2,w2=img1.shape[:2]
ch,cw=cube.shape[:2]

angle=math.pi*30/180
T1=np.float64([[1,0,-w1/2],[0,1,-h1/2],[0,0,1]])
T2=np.float64([[1,0,w1/2],[0,1,h1/2],[0,0,1]])
R=np.float64([[np.cos(angle),-np.sin(angle),0],[np.sin(angle),np.cos(angle),0],[0,0,1]])
print(T1)
print(R)
print(T2)

M=R
print(M)

imgWarp = cv2.warpPerspective(img1,M,(w1*2,h1*2))
# ~ img = cv2.warpPerspective(img,R,(w,h))
# ~ img = cv2.warpPerspective(img,T2,(w,h))
show(imgWarp)

srcPoints=np.float32([[[0,0],[w1,0],[w1,h1],[0,h1]]])
# ~ print(M@np.float64([[0],[0],[1]]))
tp=cv2.perspectiveTransform(srcPoints,M)
print(tp)
maxX,maxY=tp.max(axis=1)[0]
minX,minY=tp.min(axis=1)[0]
print(minX,maxX,minY,maxY)

# First Face

dstPoints=np.float32([[[259,129],[486,202],[486,527],[259,430]]])
M=cv2.getPerspectiveTransform(srcPoints, dstPoints)
blank=img1*0+255
squirrelPerspective=cv2.warpPerspective(img1,M,(cw,ch))
blankPerspective=cv2.warpPerspective(blank,M,(cw,ch))

kernel = np.ones((5,5),np.uint8)
erosion = cv2.erode(blankPerspective,kernel)
blankPerspective=cv2.blur(blankPerspective,(5,5))
p=blankPerspective/255.0*.8
cube=np.uint8(squirrelPerspective*p+cube*(1-p))
show(cube)

# Second Face

dstPoints=np.float32([[[486,202],[726,124],[726,430],[486,527]]])
M=cv2.getPerspectiveTransform(srcPoints, dstPoints)
blank=img2*0+255
squirrelPerspective=cv2.warpPerspective(img2,M,(cw,ch))
blankPerspective=cv2.warpPerspective(blank,M,(cw,ch))

kernel = np.ones((5,5),np.uint8)
erosion = cv2.erode(blankPerspective,kernel)
# ~ blur(blankPerspective)
blankPerspective=cv2.blur(blankPerspective,(5,5))
p=blankPerspective/255.0*.8
cube=np.uint8(squirrelPerspective*p+cube*(1-p))
show(cube)
cv2.imwrite("output.png",squirrelPerspective)
