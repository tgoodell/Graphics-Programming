#https://en.wikipedia.org/wiki/Kernel_(image_processing)

import numpy as np
import cv2

def show(img,wait=0,destroy=True):
    img=np.uint8(img)
    cv2.imshow("image",img)
    cv2.waitKey(wait)
    if destroy:
        cv2.destroyAllWindows()

def normalize(img):
    out=img*1.0
    out-=out.min()
    out/=out.max()
    out*=255.999
    return np.uint8(out)

# print(dir(cv2))
img=cv2.imread("input/image1.png")

# 1A: Develop a 3x3, 7x7, 1x7, and 1x3 Box Blur
box3x3kernel=np.float64([[1/9,1/9,1/9],
                         [1/9,1/9,1/9],
                         [1/9,1/9,1/9]])
box7x7kernel=np.float64([[1/49,1/49,1/49,1/49,1/49,1/49,1/49],
                         [1/49,1/49,1/49,1/49,1/49,1/49,1/49],
                         [1/49,1/49,1/49,1/49,1/49,1/49,1/49],
                         [1/49,1/49,1/49,1/49,1/49,1/49,1/49],
                         [1/49,1/49,1/49,1/49,1/49,1/49,1/49],
                         [1/49,1/49,1/49,1/49,1/49,1/49,1/49],
                         [1/49,1/49,1/49,1/49,1/49,1/49,1/49]])
box1x7kernel=np.float64([[1/7,1/7,1/7,1/7,1/7,1/7,1/7]])
box1x3kernel=np.float64([[1/3,1/3,1/3]])

pic_1_2_1=normalize(cv2.filter2D(img,cv2.CV_64F,box3x3kernel))
pic_1_2_2=normalize(cv2.filter2D(img,cv2.CV_64F,box7x7kernel))
pic_1_2_3=normalize(cv2.filter2D(img,cv2.CV_64F,box1x7kernel))
pic_1_2_4=normalize(cv2.filter2D(img,cv2.CV_64F,box1x3kernel))

cv2.imwrite("output/pic_1_2_1.png",pic_1_2_1)
cv2.imwrite("output/pic_1_2_2.png",pic_1_2_2)
cv2.imwrite("output/pic_1_2_3.png",pic_1_2_3)
cv2.imwrite("output/pic_1_2_4.png",pic_1_2_4)

# 1B: Using the 7x7 kernel try out all the different edge effects (Wrap, Reflect, Constant, Replicate, Default/Reflect101).

pic_1_3_1=normalize(cv2.filter2D(img,cv2.CV_64F,box7x7kernel,borderType=cv2.BORDER_REFLECT))
pic_1_3_2=normalize(cv2.filter2D(img,cv2.CV_64F,box7x7kernel,borderType=cv2.BORDER_CONSTANT))
pic_1_3_3=normalize(cv2.filter2D(img,cv2.CV_64F,box7x7kernel,borderType=cv2.BORDER_REPLICATE))
pic_1_3_4=normalize(cv2.filter2D(img,cv2.CV_64F,box7x7kernel,borderType=cv2.BORDER_REFLECT101))

cv2.imwrite("output/pic_1_3_1.png",pic_1_3_1)
cv2.imwrite("output/pic_1_3_2.png",pic_1_3_2)
cv2.imwrite("output/pic_1_3_3.png",pic_1_3_3)
cv2.imwrite("output/pic_1_3_4.png",pic_1_3_4)

# 1C: Apply the 1x7 followed by it transposed. Do a diff with the application of the full 7x7. Find max & min.
transpose=box1x7kernel.T@box1x7kernel
pic_1_4_1=normalize(cv2.filter2D(img,cv2.CV_64F,transpose))
diff=cv2.subtract(pic_1_2_3,pic_1_4_1)
cv2.imwrite("output/pic_1_4.png",diff)
print("Max:")
print(diff.max())
print("Min:")
print(diff.min())

# Gausian Blur
# 2A: Develop a 3x3, 7x7, 1x7, and 1x3 Gaussian Blur

#gauss3x3=cv2.getGaussianKernel(3,-1)

gauss3x3kernel=np.float64([[0.0625,0.125,0.0625],
                           [0.125,0.25,0.125],
                           [0.0625,0.125,0.0625]])
gauss7x7kernel=np.float64([[0.00097656,0.00341797,0.00683594,0.00878906,0.00683594,0.00341797,0.00097656],
                           [0.00341797,0.01196289,0.02392578,0.03076172,0.02392578,0.01196289,0.00341797],
                           [0.00683594,0.02392578,0.04785156,0.06152344,0.04785156,0.02392578,0.00683594],
                           [0.00878906,0.03076172,0.06152344,0.07910156,0.06152344,0.03076172,0.00878906],
                           [0.00683594,0.02392578,0.04785156,0.06152344,0.04785156,0.02392578,0.00683594],
                           [0.00341797,0.01196289,0.02392578,0.03076172,0.02392578,0.01196289,0.00341797],
                           [0.00097656,0.00341797,0.00683594,0.00878906,0.00683594,0.00341797,0.00097656]])
gauss1x7kernel=np.float64([[0.03125,0.109375,0.21875,0.28125,0.21875,0.109375,0.03125]])
gauss1x3kernel=np.float64([[0.25,0.5,0.25]])

pic_2_2_1=normalize(cv2.filter2D(img,cv2.CV_64F,gauss3x3kernel))
pic_2_2_2=normalize(cv2.filter2D(img,cv2.CV_64F,gauss7x7kernel))
pic_2_2_3=normalize(cv2.filter2D(img,cv2.CV_64F,gauss1x7kernel))
pic_2_2_4=normalize(cv2.filter2D(img,cv2.CV_64F,gauss1x3kernel))

cv2.imwrite("output/pic_2_2_1.png",pic_2_2_1)
cv2.imwrite("output/pic_2_2_2.png",pic_2_2_2)
cv2.imwrite("output/pic_2_2_3.png",pic_2_2_3)
cv2.imwrite("output/pic_2_2_4.png",pic_2_2_4)

# 2B: Using the 7x7 kernel try out all the different edge effects (Wrap, Reflect, Constant, Replicate, Default/Reflect101).
pic_2_3_1=normalize(cv2.filter2D(img,cv2.CV_64F,gauss7x7kernel,borderType=cv2.BORDER_REFLECT))
pic_2_3_2=normalize(cv2.filter2D(img,cv2.CV_64F,gauss7x7kernel,borderType=cv2.BORDER_CONSTANT))
pic_2_3_3=normalize(cv2.filter2D(img,cv2.CV_64F,gauss7x7kernel,borderType=cv2.BORDER_REPLICATE))
pic_2_3_4=normalize(cv2.filter2D(img,cv2.CV_64F,gauss7x7kernel,borderType=cv2.BORDER_REFLECT101))

cv2.imwrite("output/pic_2_3_1.png",pic_2_3_1)
cv2.imwrite("output/pic_2_3_2.png",pic_2_3_2)
cv2.imwrite("output/pic_2_3_3.png",pic_2_3_3)
cv2.imwrite("output/pic_2_3_4.png",pic_2_3_4)

# 2C: Apply the 1x7 followed by it transposed. Do a diff with the application of the full 7x7. Find max & min.
transpose=gauss1x7kernel.T@gauss1x7kernel
pic_2_4_1=normalize(cv2.filter2D(img,cv2.CV_64F,transpose))
diff=normalize(cv2.subtract(pic_2_2_3,pic_2_4_1))
cv2.imwrite("output/pic_2_4.png",diff)
print("Max:")
print(diff.max())
print("Min:")
print(diff.min())

# 3A: Make a 3x3, 5x5, 7x7, and a 9x9 sharpen kernel.

sharpen3x3kernel=np.float64([[0,-1,0],
                             [-1,5,-1],
                             [0,-1,0]])
# realsharpen3x3kernel=-1*np.ones((3,3),dtype=np.float64)
# realsharpen3x3kernel[[1],[1]]=9
# print(realsharpen3x3kernel)

#sharpen kernel: -1 all around the square of the length, i.e. -1 around a 25 in 5x5

sharpen5x5kernel=np.float64([[0,0,-1,0,0],
                             [0,-1,-2,-1,0],
                             [-1,-2,17,-2,-1],
                             [0,-1,-2,-1,0],
                             [0,0,-1,0,0]])
sharpen7x7kernel=np.float64([[0,0,-1,-2,-1,0,0],
                             [0,-1,-2,-3,-2,-1,0],
                             [-1,-2,-3,-4,-3,-2,-1],
                             [-2,-3,-4,76,-4,-3,-2],
                             [-1,-2,-3,-4,-3,-2,-1],
                             [0,-1,-2,-3,-2,-1,0],
                             [0,0,-1,-2,-1,0,0]])
sharpen9x9kernel=np.float64([[0,0,-1,-2,-3,-2,-1,0,0],
                             [0,-1,-2,-3,-4,-3,-2,-1,0],
                             [-1,-2,-3,-4,-5,-4,-3,-2,-1],
                             [-2,-3,-4,-5,162,-5,-4,-3,-2],
                             [-1,-2,-3,-4,-5,-4,-3,-2,-1],
                             [0,-1,-2,-3,-4,-3,-2,-1,0],
                             [0,0,-1,-2,-3,-2,-1,0,0]])

pic_3_2_1=normalize(cv2.filter2D(img,cv2.CV_64F,sharpen3x3kernel))
pic_3_2_2=normalize(cv2.filter2D(img,cv2.CV_64F,sharpen5x5kernel))
pic_3_2_3=normalize(cv2.filter2D(img,cv2.CV_64F,sharpen7x7kernel))
pic_3_2_4=normalize(cv2.filter2D(img,cv2.CV_64F,sharpen9x9kernel))

cv2.imwrite("output/pic_3_2_1.png",pic_3_2_1)
cv2.imwrite("output/pic_3_2_2.png",pic_3_2_2)
cv2.imwrite("output/pic_3_2_3.png",pic_3_2_3)
cv2.imwrite("output/pic_3_2_4.png",pic_3_2_4)

pic_3_2_2=normalize(cv2.filter2D(img,cv2.CV_64F,sharpen5x5kernel))
for x in range(0,10):
    pic_3_2_2 = normalize(cv2.filter2D(pic_3_2_2, cv2.CV_64F, sharpen5x5kernel))

cv2.imwrite("output/superSharpen.png",pic_3_2_2)

# Make an edge kernel and apply it.  Normalize the result. pic_4_1.png.

edgeKernel=np.float64([[-1,-1,-1],
                       [-1,8,-1],
                       [-1,-1,-1]])
pic_4_1=normalize(cv2.filter2D(img,cv2.CV_64F,edgeKernel))
cv2.imwrite("output/pic_4_1.png",pic_4_1)

# Make a diagonal edge kernel and apply it.  pic_4_2.png
dedgeKernel=np.float64([[1,0,-1],
                       [0,0,0],
                       [-1,0,1]])
pic_4_2=normalize(cv2.filter2D(img,cv2.CV_64F,dedgeKernel))
cv2.imwrite("output/pic_4_2.png",pic_4_2)


# Seward Blurs
k1=np.ones((1,25),dtype=np.float64)
k2=np.ones((25,25),dtype=np.float64)
k1/=k1.sum()
k2/=k2.sum()
out1=cv2.filter2D(img,-1,k2)
out2=cv2.filter2D(img,cv2.CV_64F,k1)
out2=cv2.filter2D(out2,cv2.CV_64F,k1.T)
out2=np.uint8(out2)
show(out1)
show(out2)
diff=out1*1.0-out2
print(diff.max(),diff.min())
show(normalize(diff))