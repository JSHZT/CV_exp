# 图像拼接

import os, sys
import os.path as osp
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

# os.chdir('pwd\\')

# ----------- 1. 基于Stitcher类 -----------

# imgPath为图片所在的文件夹相对路径
imgPath = 'sift'
imgList = os.listdir(imgPath)
imgs = []

for imgName in imgList:
    pathImg = os.path.join(imgPath, imgName)
    img = cv.imread(pathImg)
    if img is None:
        print("图片不能读取：" + imgName)
        sys.exit(-1)
    imgs.append(img)

stitcher = cv.Stitcher.create(cv.Stitcher_PANORAMA)
_result, pano = stitcher.stitch(imgs)
cv.imwrite('result/p12.png', pano)


# -------------- 2. 高斯差分 -----------------

im1 = cv.imread('sift/sift_1.jpg')
im1 = (im1[:,:,0]).astype(np.double)
sz = 7; sig = 3

# 此处补全：对im1执行高斯模糊。
# 提示：可用cv.GaussianBlur
im_gs = cv.GaussianBlur(im1, (sz,sz), sig)
im3 = im1 - im_gs
# cv.imshow("2", im3);   cv.waitKey(0)
# 此处补全：将im3的灰度值归一化至[0,255]
im3 = (cv.normalize(im3, None, alpha=0, beta=1, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)*255).astype(np.uint8)

cv.imwrite('result/s1-g.png', im1)
cv.imwrite('result/s1-gs.png', im_gs)
cv.imwrite('result/s1-cf.png', im3)


# ----------- 3. sift特征点 -----------
sift = cv.SIFT_create()
im1 = cv.imread('sift/sift_1.jpg')
im2 = cv.imread('sift/sift_2.jpg')

# 获取各个图像的特征点及sift特征向量
# 返回值kp包含sift特征的方向、位置、大小等信息
# des的shape为 (sift_num, 128)， sift_num表示图像检测到的sift特征数量
(kp1, des1) = sift.detectAndCompute(im1, None)
(kp2, des2) = sift.detectAndCompute(im2, None)

# 绘制特征点，并显示为红色圆圈
sift_1 = cv.drawKeypoints(im1, kp1, im1, color=(255, 0, 255))
sift_2 = cv.drawKeypoints(im2, kp2, im2, color=(255, 0, 255))

cv.imwrite('result/sift_1.jpg', sift_1)
cv.imwrite('result/sift_2.jpg', sift_2)


# -------------- 4. 特征点匹配 -----------------
# 特征点匹配
# K近邻算法求取在空间中距离最近的K个数据点，并将这些数据点归为一类

bf = cv.BFMatcher()
matches1 = bf.knnMatch(des1, des2, k=2)


# 调整ratio
# ratio=0.4：对于准确度要求高的匹配；
# ratio=0.6：对于匹配点数目要求比较多的匹配；
# ratio=0.5：一般情况下。
ratio1 = 0.5
good1 = []

for m1, n1 in matches1:
    # 如果最接近和次接近的比值大于一个既定的值，那么我们保留这个最接近的值，认为它和其匹配的点为good_match
    if m1.distance < ratio1 * n1.distance:
        good1.append([m1])

match_result1 = cv.drawMatchesKnn(im1, kp1, im2, kp2, good1, None, flags=2)
cv.imwrite("result/sift_1-2.png", match_result1)




























#
























#
