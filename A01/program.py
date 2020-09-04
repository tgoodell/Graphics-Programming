import numpy as np
import cv2

img=cv2.imread("input/image1.png")

def normalize(img):
    img=img-np.min(img)
    img=img/np.max(img)
    img*=255.99
    return np.uint8(img)

if np.min(img)<0 or np.max(img)>255:
    img=normalize(img)
    print("Had to normalize image")

def greyscale(img):
    b = img[:, :, 0]*0.1
    g = img[:, :, 1]*0.7
    r = img[:, :, 2]*0.2
    img=b+g+r
    return img

def blackWhite(img, threshold):
    bw = 1*img[:, :, 1]
    bw[np.uint8(bw) < threshold] = 0
    bw[np.uint8(bw) > threshold] = 255
    return bw

def desaturate(img ,percent):
    dimg = 1*img
    b = dimg[:, :, 0]
    g = dimg[:, :, 1]
    r = dimg[:, :, 2]

    b = b * percent
    b[b > 255] = 255
    b[b < 0] = 0

    g = g * percent
    g[g > 255] = 255
    g[g < 0] = 0

    r = r * percent
    r[r > 255] = 255
    r[r < 0] = 0

    dimg[:, :, 0] = b
    dimg[:, :, 1] = g
    dimg[:, :, 2] = r

    return dimg

def contrast(img, factor):
    # Set contra to a double and img for overflow reasons
    contra = np.double(img[:,:,:])

    # Actual math behind contrast
    contra[:,:,:] = 1.0*(contra[:,:,:]-128)*factor+128

    # Overflow Check
    contra[contra>255] = 255
    contra[contra<0] = 0

    return np.uint8(contra)

def tint(img, color, percent):
    timg=img[:,:,:]
    tint=127

    # Assiging bgr values
    b=timg[:, :, 0]
    g=timg[:, :, 1]
    r=timg[:, :, 2]

    # Applying tints based on color input & percent.
    if color=="blue":
        b = (1 - percent) * b + percent * tint

    if color=="green":
        g = (1 - percent) * g + percent * tint

    if color=="red":
        r = (1 - percent) * r + percent * tint

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
for i in range(0,10):
    desat=desaturate(img,0+0.1*i)
    cv2.imwrite("output/pic_2_3_" + str(i) + ".png",desat)

# 2.4 - Contrast
for i in range(0, 9):
    contImg=contrast(img,.5+i*.1)
    cv2.imwrite("output/pic_2_4_" + str(i) + ".png", contImg)

# 2.5 - Tints

# Tint image1 50% blue
blueTint=tint(img,"blue",0.5)
cv2.imwrite("output/pic_2_5_0.png", blueTint)

# Tint image1 70% green
greenTint=tint(img,"green",0.7)
cv2.imwrite("output/pic_2_5_1.png", greenTint)

# Tint image1 90% red
redTint=tint(img,"red",0.9)
cv2.imwrite("output/pic_2_5_2.png", redTint)
