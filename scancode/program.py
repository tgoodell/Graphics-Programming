import numpy as np
import cv2
import random

def show(img,wait=0,destroy=True):
    img=np.uint8(img)
    cv2.imshow("image",img)
    cv2.waitKey(wait)
    if destroy:
        cv2.destroyAllWindows()

def stats(contour):
    M = cv2.moments(contour)
    return M['m00'],(M['m10']/M['m00']),(M['m01']/M['m00'])

def decode_binary_string(s):
    return ''.join(chr(int(s[i*8:i*8+8],2)) for i in range(len(s)//8))

def drawTemplate():
    img=np.zeros(shape=[1000, 1000, 3], dtype=np.uint8)
    img[:,:,:]=255
    w,h,_=img.shape

    # Top Left Scan Code
    img[:200,:200,:]=0
    img[40:160,40:160,:]=255
    img[80:120,80:120,:]=0
    img[95:105,95:105,:]=255

    # Top Right Scan Code
    img[:200,w-200:,:]=0
    img[40:160,w-160:w-40,:]=255
    img[80:120,w-120:w-80,:]=0

    # Bottom Left
    img[h-200:,:200,:]=0
    img[h-160:h-40,40:160,:]=255
    img[h-105:h-95,95:105,:]=0

    # Bottom Right
    img[h-200:,w-200:,:]=0
    img[h-160:h-40,w-160:w-40,:]=255

    # First Row
    index=0
    while index<13:
        if index%2==0:
            coords=240+index*40
            img[:40,coords:coords+40,:]=0
        index+=1

    # Second Row
    index=0
    while index<12:
        if index%2==1:
            coords = 240 + index * 40
            img[40:80, coords:coords + 40, :] = 0
        index += 1

    # Third Row
    index = 0
    while index < 13:
        if index % 2 == 0:
            coords = 240 + index * 40
            img[80:120, coords:coords + 40, :] = 0
        index += 1

    # Fourth Row
    index = 0
    while index < 12:
        if index % 2 == 1:
            coords = 240 + index * 40
            img[120:160, coords:coords + 40, :] = 0
        index += 1

    # Fifth Row
    index = 0
    while index < 13:
        if index % 2 == 0:
            coords = 240 + index * 40
            img[160:200, coords:coords + 40, :] = 0
        index += 1

    # Bottom First Row
    index = 0
    while index < 13:
        if index % 2 == 0:
            coords = 240 + index * 40
            img[h-40:h, coords:coords + 40, :] = 0
        index += 1

    # Bottom Second Row
    index = 0
    while index < 12:
        if index % 2 == 1:
            coords = 240 + index * 40
            img[h-80:h-40, coords:coords + 40, :] = 0
        index += 1

    # Bottom Third Row
    index = 0
    while index < 13:
        if index % 2 == 0:
            coords = 240 + index * 40
            img[h - 120:h-80, coords:coords + 40, :] = 0
        index += 1

    # Bottom Fourth Row
    index = 0
    while index < 12:
        if index % 2 == 1:
            coords = 240 + index * 40
            img[h - 160:h - 120, coords:coords + 40, :] = 0
        index += 1

    # Bottom Fifth Row
    index = 0
    while index < 13:
        if index % 2 == 0:
            coords = 240 + index * 40
            img[h - 200:h - 160, coords:coords + 40, :] = 0
        index += 1

    # Left Side
    index=0
    while index<13:
        tindex=0
        if index%2==0:
            while tindex<5:
                hcoord=240+40*index
                if tindex%2==0:
                    coords=tindex*40
                    img[hcoord:hcoord+40,coords:coords+40,:]=0
                tindex+=1
        else:
            while tindex<4:
                hcoord=240+40*index
                if tindex%2==0:
                    coords=40+tindex*40
                    img[hcoord:hcoord+40,coords:coords+40,:]=0
                tindex+=1
        index+=1

    # Right Side
    index = 0
    while index < 13:
        tindex = 0
        if index % 2 == 0:
            while tindex < 5:
                hcoord = 240 + 40 * index
                if tindex % 2 == 0:
                    coords = w-200+tindex * 40
                    img[hcoord:hcoord + 40, coords:coords + 40, :] = 0
                tindex += 1
        else:
            while tindex < 4:
                hcoord = 240 + 40 * index
                if tindex % 2 == 0:
                    coords = w-160+tindex * 40
                    img[hcoord:hcoord + 40, coords:coords + 40, :] = 0
                tindex += 1
        index += 1

    index=0
    while index<13:
        bindex=0
        while bindex<13:
            val=random.randint(0,1)
            if val==0:
                img[6*40+40*index:6*40+40*index+40,6*40+40*bindex:6*40+40*bindex+40,:]=0
            bindex+=1
        index+=1

    return img

def saveCode(message):
    length='{:0>8}'.format(str(bin(len(message)))[2:])

    img=drawTemplate()

    # Encode length
    index=0
    while index<8:
        if length[index]=="0":
            img[240:280,240+index*40:280+index*40,:]=0
        else:
            img[240:280, 240 + index * 40:280 + index * 40, :] = 255
        index+=1

    index=0

    rows=len(message)
    if len(message)>13:
        rows=13


    while index<rows:
        val=message[index]
        character=''.join([bin(ord(c))[2:].rjust(8,'0') for c in val])
        bindex=0
        while bindex<8:
            if character[bindex]=="0":
                img[280+index*40:320+index*40,240+bindex*40:280+bindex*40,:] = 0
            else:
                img[280 + index * 40:320 + index * 40, 240 + bindex * 40:280 + bindex * 40, :] = 255

            bindex+=1
        index+=1

    return img

def readCode(img):
    index=0
    length=""
    while index<8:
        if np.array_equal(img[260,260+index*40,:],[0,0,0]):
            length+="0"
        else:
            length+="1"
        index+=1

    length=int(length,2)

    message=""

    bindex=0
    while bindex<length:
        character=""
        index = 0
        while index<8:
            if np.array_equal(img[300+bindex*40,260+index*40,:],[0,0,0]):
                character+="0"
            else:
                character+="1"
            index+=1

        character = decode_binary_string(character)
        message+=character
        bindex+=1

    return message

code="Tristan"
img=saveCode(code)

show(drawTemplate())
show(img)

print(readCode(img))




# print(' '.join(map(bin,bytearray(code,'utf8'))))
