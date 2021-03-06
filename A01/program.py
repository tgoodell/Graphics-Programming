import numpy as np
import cv2

img=cv2.imread("input/image1.png")

def normalize(img):
    nimg=1*img
    nimg=nimg-np.min(nimg)
    nimg=nimg/np.max(nimg)
    nimg*=255.99
    return np.uint8(nimg)

def show(img):
    simg=1*img
    if np.min(simg)<0 or np.max(simg)>255:
        simg=normalize(simg)
        print("Had to normalize image")
    return simg

def greyscale(img):
    b = img[:, :, 0]*0.1
    g = img[:, :, 1]*0.7
    r = img[:, :, 2]*0.2
    img=b+g+r
    return img

def blackWhite(img, threshold):
    bw = 1*greyscale(img)
    bw[np.uint8(bw) < threshold] = 0
    bw[np.uint8(bw) > threshold] = 255
    return bw

def desaturate(img ,percent):
    desat = 1*np.double(img[:, :, :])
    greyImg = greyscale(img)

    # Actual math behind desat
    desat[:, :, :] = (desat[:, :, :] *(1 - percent)) + (greyImg[:,:,None] * percent)
    # Overflow Check
    desat[desat > 255] = 255
    desat[desat < 0] = 0

    return desat

def contrast(img, factor):
    cimg = 1*img
    b = cimg[:, :, 0]
    g = cimg[:, :, 1]
    r = cimg[:, :, 2]

    b = (b-128.0) * factor+128
    b[b > 255] = 255
    b[b < 0] = 0

    g = (g-128.0) * factor+128
    g[g > 255] = 255
    g[g < 0] = 0

    r = (r-128.0) * factor+128
    r[r > 255] = 255
    r[r < 0] = 0

    cimg[:, :, 0] = b
    cimg[:, :, 1] = g
    cimg[:, :, 2] = r

    return cimg

def tint(img, color, percent):
    timg=img[:,:,:]
    tint=255

    # Assiging bgr values
    b=timg[:, :, 0]
    g=timg[:, :, 1]
    r=timg[:, :, 2]

    # Applying tints based on color input & percent.
    if color=="blue":
        b = 1.0*b+(255-b)*percent

    if color=="green":
        g = 1.0 * g + (255 - g) * percent

    if color=="red":
        r = 1.0 * r + (255 - r) * percent

    # Reassigning gbr values
    timg[:, :, 0] = b
    timg[:, :, 1] = g
    timg[:, :, 2] = r

    return timg

# 2.1 Greyscale Filter
pic_2_1=greyscale(img)
cv2.imwrite("output/pic_2_1.png",pic_2_1)

# 2.2a blackWhite(img, threshold=128)
pic_2_2=blackWhite(img, 128)
cv2.imwrite("output/pic_2_2.png",pic_2_2)

# 2.2b Apply blackWhite filter using threshold values from -1 to 255 in increments of 32.
for i in range(1, 9):
    pic_2_2b=blackWhite(img, (32*i)-1)
    cv2.imwrite("output/pic_2_2_" + str(i) + ".png", pic_2_2b)

pic_2_2a=blackWhite(img, -1)
cv2.imwrite("output/pic_2_2_0.png", pic_2_2a)

# 2.3 Desaturation
# Apply this filter to image1 using percent values from 0 to 1 in .1 increments
for i in range(0,11):
    desat=desaturate(img,0+0.1*i)
    cv2.imwrite("output/pic_2_3_" + str(i) + ".png",desat)

# 2.4 - Contrast
for i in range(0, 11):
    contImg=contrast(img,.5+i*.1)
    cv2.imwrite("output/pic_2_4_" + str(i) + ".png", contImg)

# 2.5 - Tints

# Tint image1 50% blue
blueTint=tint(img,"blue",0.5)
cv2.imwrite("output/pic_2_5_0.png", blueTint)

# Tint image1 70% green
greenTint=tint(img,"green",0.2)
cv2.imwrite("output/pic_2_5_1.png", greenTint)

# Tint image1 40% red
redTint=tint(img,"red",0.4)
cv2.imwrite("output/pic_2_5_2.png", redTint)
