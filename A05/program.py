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
    kernel=np.float64([[-1,0,1]])
    Ix=cv2.filter2D(img,cv2.CV_64F,kernel)
    Iy=cv2.filter2D(img,cv2.CV_64F,kernel)
    I=np.hypot(Ix,Iy)
    m=I.max()
    I[:,:margin]=m
    I[:,-margin:]=m
    return I
    
def getEnergyMap(img,repulseMask=None,attractMask=None):
    edges=getEdgeImage(img)
    if attractMask is not None:
        edges[attractMask==1]=-10
    kernel=np.ones(3,np.float64)
    for i in range(1,len(edges)):
        minAbove=cv2.erode(edges[i-1],kernel).T[0]
        edges[i]+=minAbove
    return edges


def getSeam(img,repulseMask=None,attractMask=None):
    energyMap=getEnergyMap(img,repulseMask,attractMask)
    y=len(energyMap)-1
    x=np.argmin(energyMap[y])
    seam=[(x,y)]
    while len(seam)<len(energyMap):
        x,y=seam[-1]
        newY=y-1
        newX=x+np.argmin(energyMap[newY,x-1:x+2])-1
        seam.append((newX,newY))
    return seam

def removeSeam(img,seam):
    
    output=img[:,1:]
    for (x,y),row in zip(seam,img):
        output[y,:x]=img[y,:x]
        output[y,x:]=img[y,x+1:]
    return output

def reHoriz(img,w):
    ih, iw, = img.shape[:2]
    n = 0
    seams40 = img
    gs40 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    attractMask = gs40 * 0
    attractMask[690:810, 100:200] = 1
    while n < np.abs(iw - w):
        seam = getSeam(gs40, attractMask=attractMask)
        gs40 = removeSeam(gs40, seam)
        seams40 = removeSeam(seams40, seam)
        attractMask = removeSeam(attractMask, seam)
        n += 1
    return seams40

def reVert(img,h):
    ih, iw, = img.shape[:2]
    seams40 = img
    gs40 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gs40 = cv2.rotate(gs40, cv2.ROTATE_90_COUNTERCLOCKWISE)
    seams40 = cv2.rotate(seams40, cv2.ROTATE_90_COUNTERCLOCKWISE)
    attractMask = gs40 * 0
    attractMask[690:810, 100:200] = 1
    n = 0
    while n < np.abs(ih - h):
        seam = getSeam(gs40, attractMask=attractMask)
        gs40 = removeSeam(gs40, seam)
        seams40 = removeSeam(seams40, seam)
        attractMask = removeSeam(attractMask, seam)
        n += 1
    seams40 = cv2.rotate(seams40, cv2.ROTATE_90_CLOCKWISE)
    return seams40

def retarget(img,w,h):
    seams40=reHoriz(img,w)
    seams40=reVert(seams40,h)

    return seams40


# ~ def reTarget(img,(w,h),rep,att):
#img1=cv2.resize(img1,(0,0),fx=.4,fy=.4)
#img1=cv2.GaussianBlur(img1,(37,37),-1)
#show(getEdgeImage(img1))
#show(getEnergyMap(img1))

img1=cv2.imread("input/clouds.jpg",0)
cimg1=cv2.imread("input/clouds.jpg")
attractMask=img1*0
attractMask[690:810,100:200]=1

# ~~Vertical Seaming~~

# {Energy Map}
energyMap=normalize(getEnergyMap(img1))
cv2.imwrite("output/energyMap.png",energyMap)

# {Find a Seam}
attractMask=img1*0
attractMask[690:810,100:200]=1
seam=getSeam(img1,attractMask=attractMask)
findSeam=drawSeam(cimg1,seam)
cv2.imwrite("output/highlightedSeam.png",findSeam)

# {Remove the Seam}
removedSeam=removeSeam(findSeam,seam)
cv2.imwrite("output/removedSeam.png",removedSeam)

# {Remove 40 Seams}
n=0
attractMask=img1*0
attractMask[690:810,100:200]=1
seams40=cimg1
gs40=img1
while n<=40:
    seam=getSeam(gs40,attractMask=attractMask)
    gs40=removeSeam(gs40,seam)
    seams40=removeSeam(seams40,seam)
    attractMask=removeSeam(attractMask,seam)
    n+=1
cv2.imwrite("output/seams40.png",seams40)

img1=cv2.imread("input/clouds.jpg",0)
cimg1=cv2.imread("input/clouds.jpg")

# ~~Horizontal Seaming~~

# {Highlight Seam}
himg1=cv2.rotate(img1,cv2.ROTATE_90_COUNTERCLOCKWISE)
hcimg1=cv2.rotate(cimg1,cv2.ROTATE_90_COUNTERCLOCKWISE)
attractMask=himg1*0
attractMask[690:810,100:200]=1
seam=getSeam(himg1,attractMask=attractMask)
findSeam=drawSeam(hcimg1,seam)
findSeam=cv2.rotate(findSeam,cv2.ROTATE_90_CLOCKWISE)
cv2.imwrite("output/horizontalHighlightedSeam.png",findSeam)

# {Remove 40 Seams}
himg1=cv2.rotate(img1,cv2.ROTATE_90_COUNTERCLOCKWISE)
hcimg1=cv2.rotate(cimg1,cv2.ROTATE_90_COUNTERCLOCKWISE)
attractMask=himg1*0
attractMask[690:810,100:200]=1
seam=getSeam(himg1,attractMask=attractMask)
seams40=hcimg1
gs40=himg1
n=0
while n<=40:
    seam=getSeam(gs40,attractMask=attractMask)
    gs40=removeSeam(gs40,seam)
    seams40=removeSeam(seams40,seam)
    attractMask=removeSeam(attractMask,seam)
    n+=1
seams40=cv2.rotate(seams40,cv2.ROTATE_90_CLOCKWISE)
cv2.imwrite("output/horizontalSeams40.png",seams40)

cv2.imwrite("output/640x480.png",retarget(cimg1,640,480))
cv2.imwrite("output/640x640.png",retarget(cimg1,640,640))
cv2.imwrite("output/320x240.png",retarget(cimg1,320,240))
cv2.imwrite("output/320x320.png",retarget(cimg1,320,320))
