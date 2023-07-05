import os, sys
import os.path as osp
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

def cat_picture(imgs, type='Stitcher'):
    '''
    imgs:images list(BGR)
    type:method name
    '''
    if type == 'Stitcher':
        stitcher = cv.Stitcher_create()
        _result, pano = stitcher.stitch(imgs)
        return pano
    
def get_GaussPyramid(img, high, sigs:list, return_difference:bool=False)->dict:
    GP = {}
    src = (img[:,:,0]).astype(np.double)
    for i in range(high):
        layer = [src]
        for sig in sigs:
            sz = int(2 * np.ceil(3 * sig) + 1)
            gs_ = cv.GaussianBlur(src, (sz,sz), sig)
            layer.append(gs_)
        GP[str(i)] = layer.copy()
        src = cv.pyrDown(layer[3])
    if return_difference:
        GPD={}
        for i in range(high):
            layer = GP[str(i)]
            layer_d = []
            for j in range(len(layer)-1):
                layer_d.append(layer[j]-layer[j+1])
            GPD[str(i)] = layer_d.copy()
        return{"GP":GP,"GPD":GPD}
    return {"GP":GP}
        
if __name__ == '__main__':
    #1.使用Stitcher类，实现4张或以上图像的拼接
    imgPath = 'sift'
    imgList = os.listdir(imgPath)
    imgs = []
    for imgName in imgList:
        pathImg = os.path.join(imgPath, imgName)
        img = cv.imread(pathImg)
        resized_image = cv.resize(img, (640, 480))
        imgs.append(img.copy())
    pano = cat_picture(imgs)
    cv.imwrite('result/T1.png', pano)
    
    #2.计算一幅图像的高斯差分金字塔。请展示金字塔每一层中 最底层 图像的 高斯模糊和高斯差分效果
    img = cv.imread('T2.jpg')
    k = np.sqrt(2)
    sigma = 1.6
    high = 5
    sigs = []
    for j in range(1, 6):
        sigs.append(sigma * (k ** j))
    opdict = get_GaussPyramid(img, high, sigs, True)
    for i in range(high):
        img_bur = opdict['GP'][str(i)][0]
        img_bur = (cv.normalize(img_bur, None, alpha=0, beta=1, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)*255).astype(np.uint8)
        img_d = opdict['GPD'][str(i)][0]
        img_d = (cv.normalize(img_d, None, alpha=0, beta=1, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)*255).astype(np.uint8)
        cv.imwrite(f'result/bur_layer{i+1}.png', img_bur)
        cv.imwrite(f'result/d_layer{i+1}.png', img_d)
    
    #3.将找到的极值点以彩色点标注在原始图像上（来自金字塔第一层的极值点-红色，第二层-绿色，第三层-蓝色，第四层-紫色，第五层-黄色）
    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 0, 255), (0, 255, 255)]
    sift = cv.SIFT_create()
    img = cv.imread('T2.jpg')
    for i in range(high):
        # im = (cv.normalize(opdict['GP'][str(i)][0], None, alpha=0, beta=1, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)*255).astype(np.uint8)
        im = cv.imread('result/'+f'bur_layer{i+1}.png')
        im = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
        (tp1, des1) = sift.detectAndCompute(im, None)
        sift_1 = cv.drawKeypoints(img, tp1, img, color=colors[i])
    cv.imwrite(f'extreme_{i}.png', sift_1)
    #4.计算能够相互匹配的 特征点之间的平均欧式距离
    
    def compute_average_euclidean_distance(matches):
        total_distance = 0
        num_matches = len(matches)
        for match in matches:
            total_distance += match.distance
        return total_distance / num_matches
    
    sift = cv.SIFT_create()
    image1 = cv.imread('sift/sift_1.jpg')
    image2 = cv.imread('sift/sift_2.jpg')

    gray_image1 = cv.cvtColor(image1, cv.COLOR_BGR2GRAY)
    gray_image2 = cv.cvtColor(image2, cv.COLOR_BGR2GRAY)
    keypoints1, descriptors1 = sift.detectAndCompute(gray_image1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(gray_image2, None)
    bf = cv.BFMatcher(cv.NORM_L2, crossCheck=True)
    matches = bf.match(descriptors1, descriptors2)
    average_distance = compute_average_euclidean_distance(matches)
    print(average_distance)