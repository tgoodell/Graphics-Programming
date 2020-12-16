import numpy as np
import cv2
import struct #secret sauce   bytes->number->bytes
#bytes to thing :  UNPACK
#thing to bytes :  PACK
#https://docs.python.org/3/library/struct.html

def blackWhite(img, threshold):
    bw = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    bw[np.uint8(bw) < threshold] = 0
    bw[np.uint8(bw) > threshold] = 255
    return bw

def getWhitePixelCount(pixels,startIndex,count=0):
    counter=count
    index=startIndex
    if pixels[index]=="0":
        counter+=1
        index+=1
        getWhitePixelCount(pixels, index, counter)
    else:
        return counter

    return -2

def getBlackPixelCount(pixels,startIndex,count=0):
    counter=count
    index=startIndex
    if pixels[index]=="0":
        counter+=1
        index+=1
        getBlackPixelCount(pixels, index, counter)
    else:
        return counter

    return -1

def save(img,filename):
    w,h=img.shape
    f= open(filename+".tzg","wb")
    f.write(b"TZG")
    aSeriesOfBytes=struct.pack("<2I",w,h)
    f.write(aSeriesOfBytes)
    pixels=""
    for row in img:
        for pixel in row:
            pixels+="1" if pixel>128 else "0"

    index=0
    counter=0
    turn=0
    while index<len(pixels):
        counter=0
        if pixels[index]=="0":
            while index<len(pixels)-1 and pixels[index+1]=="0":
                counter+=1
                index+=1
            print("Black: " + str(counter))
        elif pixels[index]=="1":
            while index<len(pixels)-1 and pixels[index + 1] == "1":
                counter += 1
                index += 1
            print("White: " + str(counter))

        index += 1


    # print(pixels)

    padSize=(8-len(pixels))%8
    pixels+="0"*padSize
    for i in range(0,len(pixels),8):
        x=int(pixels[i:i+8],2)
        f.write(struct.pack("<B",x))
    f.close()

def read(filename):
    f= open("%s.tzg"%filename,"rb")
    x=f.read(3)
    if x!=b"TZG":
        print("invalid file")
    w,h=struct.unpack("<2I",f.read(8))
    
    s=""
    while 1:
        b=f.read(1)
        if not b:
            break
        v=struct.unpack("<B",b)[0]
        s+=bin(v)[2:].zfill(8)

    m=np.array(list(s))
    img=np.uint8(np.reshape(m[:w*h],(w,h))=="1")*255
    return img
    # ~ cv2.imwrite("test.png",img)
    # ~ print(img)
#
img = cv2.imread("../flago.png")
img=blackWhite(img,127)
save(img,"canada")
#
# cv2.imwrite("out2.png",read("xkcd-test"))
