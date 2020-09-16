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
book=cv2.imread("input/book.png")
h1,w1=img1.shape[:2]
h2,w2=img1.shape[:2]
ch,cw=cube.shape[:2]
bh,bw=cube.shape[:2]

angle=math.pi*30/180
T1=np.float64([[1,0,-w1/2],[0,1,-h1/2],[0,0,1]])
T2=np.float64([[1,0,w1/2],[0,1,h1/2],[0,0,1]])
T4=np.float64([[1,0,100],[-100,1,0],[0,0,1]])
R=np.float64([[np.cos(angle),-np.sin(angle),0],[np.sin(angle),np.cos(angle),0],[0,0,1]])

# 1 - Mirror Across Y-Axis
mirror=np.float64([[-1,0,w1],[0,1,0],[0,0,1]])
trans1 = cv2.warpPerspective(img1,mirror,(w1,h1))
cv2.imwrite("output/pic_1_1.png",trans1)
show(trans1)

# 2 - 30 Degrees Clockwise About Lower Right Corner
angle=math.pi*30/180
flipy=np.float64([[-1,0,w1],[0,1,0],[0,0,1]])
rotate30=np.float64([[-np.cos(angle),np.sin(angle),0],[np.sin(angle),np.cos(angle),0],[0,0,1]])
left=np.float64([[1,0,300],[0,1,0],[0,0,1]])
final=flipy@rotate30@flipy@left
final = cv2.warpPerspective(img1,final,(w1,h1))
cv2.imwrite("output/pic_1_2.png",final)
show(final)

# 3 - 30 Degree Rotation w/o margins
angle=math.pi*30/180
ninety=math.pi*90/180
side1=h1*np.sin(ninety-angle)/np.sin(ninety)
side2=h1*np.sin(angle)/np.sin(ninety)
side3=w1*np.sin(ninety-angle)/np.sin(ninety)
side4=w1*np.sin(angle)/np.sin(ninety)

newHeight=np.uint64(side1+side4)
newWidth=np.uint64(side2+side3)

flipy=np.float64([[-1,0,w1],[0,1,0],[0,0,1]])
rotate30=np.float64([[-np.cos(angle),np.sin(angle),0],[np.sin(angle),np.cos(angle),0],[0,0,1]])
left=np.float64([[1,0,300],[0,1,0],[0,0,1]])
final=flipy@rotate30@flipy@left
final = cv2.warpPerspective(img1,final,dsize=(newHeight,newWidth))

show(final)
cv2.imwrite("output/pic_1_3.png",final)


pts1 = np.float32([[124,455],[547,349],[883,781],[393,979]])
pts2 = np.float32([[0,0],[bw,0],[bw,bh],[0,bh]])

# Apply Perspective Transform Algorithm
M = cv2.getPerspectiveTransform(pts1, pts2)
result=cv2.warpPerspective(book,M,(w1,h1))

# ~ print(M@np.float64([[0],[0],[1]]))
#tp=cv2.perspectiveTransform(srcPoints,dstPoints)
show(result)
#print(tp)
#maxX,maxY=tp.max(axis=1)[0]
#minX,minY=tp.min(axis=1)[0]
#print(minX,maxX,minY,maxY)

# Cube w/ Faces
# First Face
srcPoints=np.float32([[[0,0],[w1,0],[w1,h1],[0,h1]]])

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

# Second Face
img2square=img2[:h1,:w1,:]

dstPoints=np.float32([[[486,202],[726,124],[726,430],[486,527]]])
M=cv2.getPerspectiveTransform(srcPoints, dstPoints)
blank=img2square*0+255
squirrelPerspective=cv2.warpPerspective(img2square,M,(cw,ch))
blankPerspective=cv2.warpPerspective(blank,M,(cw,ch))

kernel = np.ones((5,5),np.uint8)
erosion = cv2.erode(blankPerspective,kernel)
blankPerspective=cv2.blur(blankPerspective,(5,5))
p=blankPerspective/255.0*.8
cube=np.uint8(squirrelPerspective*p+cube*(1-p))
show(cube)
cv2.imwrite("output/pic_1_4.png",cube)
