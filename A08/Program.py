import numpy as np
import cv2
import math
import zlib
import struct  # secret sauce   bytes->number->bytes
from PIL import Image

DEBUG=True


# bytes to thing :  UNPACK
# thing to bytes :  PACK
# https://docs.python.org/3/library/struct.html

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

def blackWhite(img, threshold):
    bw = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    bw[np.uint8(bw) < threshold] = 0
    bw[np.uint8(bw) > threshold] = 255
    return bw

def save(img, filename):
    w, h = img.shape
    f = open(filename + ".tgf", "wb")
    f.write(b"TGF")
    aSeriesOfBytes = struct.pack("<2I", w, h)
    f.write(aSeriesOfBytes)
    pixels = ""
    for row in img:
        for pixel in row:
            pixels += "1" if pixel > 128 else "0"

    print(len(pixels))


    # print(len(pixels))
    # index=0
    # npixels=""
    # while index<len(pixels):
    #     test=len(pixels)-index
    #     if test>3:
    #         if pixels[index]==pixels[index+1] and pixels[index]==pixels[index+2]:
    #             if pixels[index]=="0":
    #                 npixels+="2"
    #             elif pixels[index]=="1":
    #                 npixels+="3"
    #             index+=3
    #         elif pixels[index]=="0" or pixels[index]=="1":
    #             npixels+=pixels[index]
    #             index+=1
    #     elif pixels[index] == "0" or pixels[index] == "1":
    #         npixels += pixels[index]
    #         index += 1

    padSize = (8 - len(pixels)) % 8
    pixels += "0" * padSize

    compressed = zlib.compress(bytes(pixels,"utf-8"))

    text_file = open("temp3.tzg", "w")
    text_file.write(str(compressed))
    text_file.close()
    f.close()

def read(filename):
    f = open("%s.tzg" % filename, "rb")

    s = ""
    index=0
    while 1:
        b = f.read(1)
        if not b:
            break
        v = struct.unpack("<B", b)[0]
        s += str(v)

    m = np.array(list(s))
    uncompressed=zlib.decompress()
    text_file = open("temp4.txt", "w")
    text_file.write(str(uncompressed))
    text_file.close()

    f = open("temp4.txt", "rb")
    x = f.read(3)
    if x != b"TGF":
        print("invalid file")
    w, h = struct.unpack("<2I", f.read(8))




    # s = ""
    # index=0
    # while 1:
    #     b = f.read(1)
    #     if not b:
    #         break
    #     v = struct.unpack("<B", b)[0]
    #     s += str(v)
    #
    # m = np.array(list(s))
    # print(len(m))
    #
    # pixels=""
    # index=0
    # while index<len(m):
    #     if m[index]=="2":
    #         pixels+="000"
    #     elif m[index]=="3":
    #         pixels+="111"
    #     elif m[index]=="0":
    #         pixels+="0"
    #     elif m[index]=="1":
    #         pixels+="1"
    #     index+=1
    #
    # text_file = open("temp.txt", "w")
    # text_file.write(pixels)
    # text_file.close()
    #
    # f = open("temp.txt", "rb")
    #
    # s = ""
    # while 1:
    #     b = f.read(1)
    #     if not b:
    #         break
    #     v = struct.unpack("<B", b)[0]
    #     s += bin(v)[2:].zfill(8)
    #
    #
    #
    # m = np.array(list(s))
    # img = np.uint8(np.reshape(m[:w * h], (w, h)) == "1") * 255
    # cv2.imwrite("out2.png", img)


# img = cv2.imread("flago.png")
# img=blackWhite(img,127)
# save(img,"canada")
#
# read("temp3")

# save(img, "test2")
# cv2.imwrite("out2.png", read("test2"))
