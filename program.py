import numpy as np
import cv2

# Switch red and green channels in image1
image1=cv2.imread("input/image1.png")
red=image1[:,:,2]
green=image1[:,:,1]
image1_gr_switch=image1[:,:,:]
image1_gr_switch[:,:,1]=red
image1_gr_switch[:,:,2]=green

cv2.imwrite("output/pic_1_a.png",image1_gr_switch)

# Extract blue channel to become grayscale in image2
image2=cv2.imread("input/image2.png")
blue=image2[:,:,0]
cv2.imwrite("output/pic_1_b.png",blue)

# Invert the green channel of image1 without touching the other channels
image1=cv2.imread("input/image1.png")
red=image1[:,:,2]
blue=image1[:,:,0]
green=255-image1[:,:,2]
image1_gi_switch=image1[:,:,:]
image1_gi_switch[:,:,1]=red
image1_gi_switch[:,:,0]=blue
image1_gi_switch[:,:,2]=green
cv2.imwrite("output/pic_1_c.png",image1_gi_switch)

# Add 100 to each channel of image2
image2=cv2.imread("input/image2.png")
image2[image2>255-100]=255
image2[image2<255]+=100
cv2.imwrite("output/pic_1_d.png",image2)

# Select middle 100x100 region and replace green channel with 255 with image2
image2=cv2.imread("input/image2.png")
h,w=image2.shape[:2]
red=image2[:,:,2]
blue=image2[:,:,0]
nimage2=image2
nimage2[h*103//256:h*153//256,w*71//192:w*121//192,1]=255
nimage2[:,:,0]=blue
nimage2[:,:,2]=red

# Paste center 100x100 of image2 onto image1
image1=cv2.imread("input/image1.png")
image2=cv2.imread("input/image2.png")
paste=image2[h//2-50:h//2+50,w//2-50:w//2+50]

h,w=image1.shape[:2]
image1[h//2-50:h//2+50,w//2-50:w//2+50]=paste
cv2.imwrite("output/pic_2_b.png",image1)

# Stats
image1=cv2.imread("input/image1.png")
print("Stats {image1}:")
print(image1.size)
print(image1.min())
print(image1.max())
print(image1.std())
print(image1.mean())
print("-----\n")

# Define star
def star(img,r,center=(0,0),n=5,cycles=2,color=255,rotation=0):
    xc,yc=center
    h,w,*_=img.shape
    Y,X=np.mgrid[:h,:w]
    X-=xc
    Y-=yc
    d=np.hypot(X,Y)
    theta=np.arctan2(Y,X)+np.pi/2-rotation
    theta=theta%(np.pi*2/n)
    theta=np.minimum(theta,np.pi*2/n-theta)
    XP=d*np.cos(theta)
    YP=d*np.sin(theta)
    phi=np.arctan2(YP,XP-r)
    tipAngle=np.pi-2*np.pi*cycles/n
    img[phi>np.pi-tipAngle/2]=color

# cv2.imread("filename.png")
img=np.zeros((400,800,3),dtype=np.uint8)
h,w=img.shape[:2]

# [blue, green, red]
white=(255,255,255)
red=(43,20,204)
blue=(125,36,0)
green=(0,255,0)

# img[:h//3]=white
# img[h//3:h//3*2]=blue
# img[h//3*2:]=red

# Background Blue
img[::]=blue

# Diagonal White Bar from Left
sp0=(0,0)
ep0=(w,h)
thck0=w*6//58
bar0 = cv2.line(img, sp0, ep0, white, thck0)

# Diagonal White Bar from Right
sp1=(0,h)
ep1=(w,0)
thck1=w*6//58
bar1 = cv2.line(img, sp1, ep1, white, thck1)

# First Diagonal Red Bar CW
sp2=(0,h*1//28)
ep2=(w//2,h//2+h*1//28)
thck2=w*2//58
bar2 = cv2.line(img, sp2, ep2, red, thck2)

# Second Diagonal Red Bar CW
sp3=(w-w*1//28,0)
ep3=(w//2-w*1//28,h//2)
thck3=w*2//58
bar3 = cv2.line(img, sp3, ep3, red, thck3)

# Third Diagonal Red Bar CW
sp4=(w,h-h*1//28)
ep4=(w//2+w*1//28,h//2)
thck4=w*2//58
bar4 = cv2.line(img, sp4, ep4, red, thck4)

# Third Diagonal Red Bar CW
sp5=(0,h+h*1//28)
ep5=(w//2+w*1//28,h//2)
thck5=w*2//58
bar5 = cv2.line(img, sp5, ep5, red, thck5)

# Horizontal White Bar
img[h*9//28:h*19//28]=white

# Vertical White Bar
sp6=(w//2,0)
ep6=(w//2,h)
thck6=w*10//58
bar6 = cv2.line(img, sp6, ep6, white, thck6)

# Horizontal Red Middle Bar
img[h*11//28:h*17//28]=red

# Vertical Red Bar
sp7=(w//2,0)
ep7=(w//2,h)
thck7=w*6//58
bar7 = cv2.line(img, sp7, ep7, red, thck7)

# Create actual flag image
flag=np.zeros((800,1600,3),dtype=np.uint8)

# Blue background
flag[::]=blue

# import union jack
flag[:img.shape[0],:img.shape[1]] += img[:,:]
flag[:img.shape[0],:img.shape[1]]  = img[:,:]
flag[:img.shape[0],:img.shape[1]]  = img

# Redefine w, h
h,w=flag.shape[:2]

# Stars
o=60/80
# Top Star
star(flag,o*w*13//240,(w-w*1//4,h*1//5),color=white)
star(flag,o*w*1//30,(w-w*1//4,h*1//5),color=red)

# Mid Star Left
star(flag,o*w*13//240,(w*380//599,h*102//236),color=white)
star(flag,o*w*1//30,(w*380//599,h*102//236),color=red)

# Mid Star Right
star(flag,o*w*12//240,(w*509//599,h*111//299),color=white)
star(flag,o*w*1//34,(w*509//599,h*111//299),color=red)

# Bottom Star
star(flag,o*w*1//16,(w-w*1//4,h-h*1//5),color=white)
star(flag,o*w*5/120,(w-w*1//4,h-h*1//5),color=red)


cv2.imwrite("output/unionjack.png",img)
cv2.imwrite("output/pic_4_a.png",flag)
print("Image created")

flag=cv2.imread("input/flag.png")
pic4a=cv2.imread("output/pic_4_a.png")

def normalize(img):
    img=img-np.min(img) #the smallest value is 0
    img=img/np.max(img) #the largest value is 1
    img*=255.99
    return np.uint8(img)

normal=(flag*1.0-pic4a)
normal=normalize(normal)
cv2.imwrite("output/pic_4_b.png",normal)
print("Flags normalized")