import numpy as np
import cv2
import time
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

def getTargetSize(points,minx,miny,combinePano):
    maxx = np.max(points[0, :, 0])
    maxy = np.max(points[0, :, 1])
    target_size = (int(maxx - minx), int(maxy - miny))
    if combinePano==1:
        target_size=(int(2088), int(1426))
    return target_size

def panorizor(img1,img2,combinePano,blur):
    orb = cv2.ORB_create()
    kp1 = orb.detect(img1)
    kp2 = orb.detect(img2)
    kp1, des1 = orb.compute(img1, kp1)
    kp2, des2 = orb.compute(img2, kp2)
    bf = cv2.BFMatcher()

    matches = bf.knnMatch(des1, des2, k=2)
    good = []
    for m, n in matches:
        if m.distance < 0.9 * n.distance:
            print(des1[m.queryIdx])
            print(des2[m.trainIdx])
            print()
            good.append(m)

    img3 = cv2.drawMatches(img1, kp1, img2, kp2, good, None)
    cv2.imwrite("output/matches.png", img3)
    #show(img3)

    points1 = []
    points2 = []
    for m in good:
        pt1 = kp1[m.queryIdx].pt
        pt2 = kp2[m.trainIdx].pt
        points1.append(pt1)
        points2.append(pt2)

    H, _ = cv2.findHomography(np.float32(points2), np.float32(points1), cv2.RANSAC, 5.0)
    print(H)

    #show(img1)
    #show(cv2.warpPerspective(img2, H, (0, 0)))
    HI = np.linalg.inv(H)
    #show(img2)
    #show(cv2.warpPerspective(img1, HI, (0, 0)))

    diff = normalize(img2 * 1.0 - cv2.warpPerspective(img1, HI, (0, 0)))
    #show(diff)
    h, w, = img1.shape[:2]
    points = np.float64([[[0, 0], [w - 1, 0], [0, h - 1], [w - 1, h - 1]]])
    points = np.hstack((points, cv2.perspectiveTransform(points, H)))

    minx = np.min(points[0, :, 0])
    miny = np.min(points[0, :, 1])
    target_size=getTargetSize(points,minx,miny,combinePano)
    translate = np.float64([[1, 0, -minx], [0, 1, -miny], [0, 0, 1]])
    out1 = cv2.warpPerspective(img2, translate.dot(H), target_size)
    mask = cv2.warpPerspective(np.uint8(img2 * 0 + 255), translate.dot(H), target_size)
    out2 = cv2.warpPerspective(img1, translate, target_size)
    diff = normalize((out1 - 1.0 * out2) ** 2)
    cv2.imwrite("output/diff.png", diff)
    cv2.imwrite("output/out1.png", out1)
    cv2.imwrite("output/out2.png", out2)

    if blur==1:
        # erode mask
        #show(mask)
        transition_zone = 129
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (transition_zone, transition_zone))
        mask = cv2.erode(mask, kernel, iterations=1)
        #show(mask)
        # blur mask
        mask = cv2.blur(mask, (transition_zone, transition_zone))

        #show(mask)
        out = mask / 255 * out1 + (1 - mask / 255) * out2
    else:
        out = mask / 255 * out1 + (1 - mask / 255) * out2

    #show(out)
    cv2.imwrite("output/out.png", out)
    return out



img1=cv2.imread("input/image1.jpg")
img2=cv2.imread("input/image2.jpg")
img3=cv2.imread("input/image3.jpg")

pano1=panorizor(img1,img2,1,1)
pano2=panorizor(img1,img3,1,1)

cv2.imwrite("output/pano1.png", pano1)
cv2.imwrite("output/pano2.png", pano2)

img1=cv2.imread("output/pano1.png")
img2=cv2.imread("output/pano2.png")
panoTotal=panorizor(img1,img2,0,1)

cv2.imwrite("output/panoTotal.png", panoTotal)

# ~ for d in des1:
    # ~ print(d)

#BF: BRUTE FORCE
#KNN: k-Nearest Neighbors


# ~ good=[m[0] for m in matches]
