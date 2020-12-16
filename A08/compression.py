# Grade based on how much more compressed you can make an image than just making it black and white.
# Make your own TLA, three letter acronym, for your own file extension.
# Above 80 - Work in binary
# Hex Editor!
# Make sure you have unique magic nums - numbers at beginning of unique file type so you can recognize your files.
# 5BYTES SWARD
# 4BYTES WIDTH
# 4BYTES HEIGHT
# PIXELS STORED 1 BYTE EACH

# Beat 1 bit/pixel - 90%
# write the string instead of going to the binary - 80%
# Can do lossy. Max 0.2% different.
# Run Length Encoding
# Variable Length Encoding

import numpy as np
import cv2
import struct # secret sauce    bytes -> nums -> bytes

# def save(img,filename):
#     w,h=img.shape
#     f=open("filename")

img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

img=cv2.imread("flago.png",0)
w,h=img.shape
print(img)
f=open("test.swd","wb")

# x=f.read
# if x!="SWARD":
#     print("invalid file")

w,h=struct.unpack("<2I",f.read(8))
print(w,h)
pixels=[]
while 1:
    b=f.read(1)
    if not b:
        break
    pixels.appen(struct.unpack("<B",b))

img=np.reshape(pixels,(w,h))
cv2.imwrite(img,"test.png")

f.write(b"SWARD")
aSeriesOfBytes=struct.pack("<2I",w,h)
f.write(aSeriesOfBytes)

pixels=""
for row in img:
    for pixel in row:
        # f.write(struct.pack("<B",pixel))
        pixels+="1" if pixel>128 else "0"

padSize=(8-len(pixels))%8
pixels+="0"*padSize
for i in range(0,len(pixels),8):
    x=int(pixels[i:i+8],2)
    f.write(struct.pack("B",x))

f.close()