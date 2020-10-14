import numpy as np
import cv2
import math
DEBUG=True

def show(img,title="image",wait=True):
    d=max(img.shape[:2])
    if d>1000:
        step=int(math.ceil(d/1000))
        img=img[::step,::step]
    if not DEBUG:
        return
    if np.all(0<=img) and np.all(img<256):
        cv2.imshow(title,np.uint8(img))
    else:
        print("normalized version")
        cv2.imshow(title,normalize(img))
    if wait:
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        cv2.waitKey(1)

def normalize(img):
    img_copy=img*1.0
    img_copy-=np.min(img_copy)
    img_copy/=np.max(img_copy)
    img_copy*=255.9999
    return np.uint8(img_copy)

def drawSeam(img,seam,color=0):
    copyImg=img*1
    for x,y in seam:
        copyImg[y,x]=color
    return copyImg

def getEdgeImage(img,margin=10):
    kernel = np.float64([[-1, 0, 1]])
    Ix = cv2.filter2D(img, -1, cv2.CV_64F, kernel)
    Iy = cv2.filter2D(img, -1, cv2.CV_64F, kernel)
    I = np.hypot(Ix, Iy)
    m=I.max()
    I[:,:margin]=m
    I[:,-margin:]=m
    return I

def getEnergyMap(img):
    edges=getEdgeImage(img)
    kernel=np.ones(3,np.float64)
    for i in range(1,len(edges)):
        minAbove=cv2.erode(edges[i-1],kernel).T[0]
        edges[i] += minAbove[:, 0]
    return edges

def getSeam(img):
    energyMap=getEnergyMap(img)
    y=len(energyMap)-1
    x = np.argmin(energyMap[y])
    seam = [(x, y)]
    while len(seam) < len(energyMap):
        x, y = seam[-1]
        newY = y - 1
        newX = x + np.argmin(energyMap[newY, x - 1:x + 2]) - 1
        seam.append((newX, newY))

    return seam

def removeSeam(img):
    seam=getSeam(img)
    output=img[:,1:]
    for (x,y),row in zip(seam,img):
        output[y,:x]=img[y:x]
        output[y,x:]=img[y,x+1:]
    return output

img1=cv2.imread("input/lakeOne.jpg")
img2=cv2.imread("input/lakeTwo.jpg")


show(getEnergyMap(img1))
show(getEdgeImage(img1))
img3=removeSeam(img1)



diff=normalize(diff)*0
for x,y in seam:
    diff[x,y]=255
show(diff)

cv2.floodFill(diff,None,(0,0),255)
show(diff)
cv2.imwrite("diffseam.png",diff)
diff=diff/255.0
diff=cv2.getGaussianKernel(diff,(7,7),-1)
out=out1*diff+out2*(1-diff)
show(out)
cv2.imwrite("compseam.ong",out)
out=drawSeam(out,seam)
show(out)

# diff=normalize(img2*1.0-cv2.warpPerspective(img1,HI,(0,0)))
# show(diff)
# ~ h,w,=img1.shape[:2]
# ~ points=np.float64([[[0,0],[w-1,0],[0,h-1],[w-1,h-1]]])
# ~ points=np.hstack((points,cv2.perspectiveTransform(points, H)))

# ~ minx=np.min(points[0,:,0])
# ~ maxx=np.max(points[0,:,0])
# ~ miny=np.min(points[0,:,1])
# ~ maxy=np.max(points[0,:,1])
# ~ target_size=(int(maxx-minx),int(maxy-miny))
# ~ translate=np.float64([[1,0,-minx],[0,1,-miny],[0,0,1]])
# ~ out1=cv2.warpPerspective(img2,translate.dot(H),target_size)
# ~ mask=cv2.warpPerspective(np.uint8(img2*0+255),translate.dot(H),target_size)
# ~ out2=cv2.warpPerspective(img1,translate,target_size)
# ~ diff=normalize((out1-1.0*out2)**2)
# ~ cv2.imwrite("diff.png",diff)
# ~ cv2.imwrite("out1.png",out1)
# ~ cv2.imwrite("out2.png",out2)

# ~ #erode mask
# ~ show(mask)
# ~ transition_zone=129
# ~ kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(transition_zone,transition_zone))
# ~ mask = cv2.erode(mask,kernel,iterations = 1)
# ~ show(mask)
# ~ #blur mask
# ~ mask=cv2.blur(mask,(transition_zone,transition_zone))


# ~ show(mask)
# ~ out=mask/255*out1+(1-mask/255)*out2


# ~ show(out)
# ~ cv2.imwrite("out.png",out)

