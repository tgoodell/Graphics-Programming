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

def makePatch(r):
    patch=cv2.imread("patch.png",0)
    patch=cv2.resize(patch,(r*2,r*2))
    patch=cv2.rotate(patch,cv2.ROTATE_180)

    patch=cv2.Canny(patch,10,50)
    return patch

def idVotes(cimg,blur,r):
    img = cv2.cvtColor(cimg, cv2.COLOR_BGR2GRAY)
    img = cv2.blur(img, (blur, blur))
    h, w = img.shape
    leaf = makePatch(r)
    votes = np.zeros((h + 2 * r, w + 2 * r), dtype=np.float64)
    edges = cv2.Canny(img, 10, 40)
    y, x = np.where(edges > 128)
    for i, j in zip(x, y):
        votes[j:j + r * 2, i:i + r * 2] += leaf

    return votes

def idMax(cimg,blur):
    max=0
    maxCoords=(0,0)
    r=10
    while(r<=1000):
        votes = idVotes(cimg, blur, r)
        maybeMaxVals=cv2.minMaxLoc(votes)
        maybeMax=maybeMaxVals[1]
        if maybeMax>max:
            max=maybeMax
            maxCoords=maybeMaxVals[3]
        r+=100

    return maxCoords

def idLeaf(img,blur):
    cimg=img
    img = cv2.cvtColor(cimg, cv2.COLOR_BGR2GRAY)
    h, w = img.shape
    coords=idMax(cimg,blur)
    r=10

    target=cv2.resize(cimg,(w+2*r,h+2*r))
    cv2.circle(target,coords,r*2,(0,255,0),20)

    return target

cimg=cv2.imread("input/flago.png")
show(idLeaf(cimg,7))

# idLeaf(cimg,7,200)
# idLeaf(cimg,7,220)
# idLeaf(cimg,7,240)
# idLeaf(cimg,7,300)

# cimg=cv2.imread("input/flago.png")
# fflagVotes=normalize(idVotes(cimg,7,220))
# cv2.imwrite("output/fflagVotes.png",fflagVotes)
# fflagOut=idLeaf(cimg,7,220)
# cv2.imwrite("output/fflagOut.png",fflagOut)
#
# cimg=cv2.imread("input/flag1.jpeg")
# flyFlagVotes=normalize(idVotes(cimg,7,100))
# cv2.imwrite("output/flyFlagVotes.png",flyFlagVotes)
# flyFlagOut=idLeaf(cimg,7,100)
# cv2.imwrite("output/flyFlagOut.png",flyFlagOut)
#
# cimg=cv2.imread("input/ornament.jpeg")
# ornamentVotes=normalize(idVotes(cimg,7,150))
# cv2.imwrite("output/ornamentVotes.png",ornamentVotes)
# ornamentOut=idLeaf(cimg,7,150)
# cv2.imwrite("output/ornamentOut.png",ornamentOut)
#
# cimg=cv2.imread("input/coin.jpeg")
# coinVotes=normalize(idVotes(cimg,7,150))
# cv2.imwrite("output/coinVotes.png",coinVotes)
# coinOut=idLeaf(cimg,7,150)
# cv2.imwrite("output/coinOut.png",coinOut)
#
# cimg=cv2.imread("input/blanket.jpg")
# blanketVotes=normalize(idVotes(cimg,25,190))
# cv2.imwrite("output/blanketVotes.png",blanketVotes)
# blanketOut=idLeaf(cimg,25,190)
# cv2.imwrite("output/blanketOut.png",blanketOut)