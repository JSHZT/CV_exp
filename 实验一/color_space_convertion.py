import numpy as np
import cv2
import math
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from util import convert

def RGB2YUV_enhance(img, lightness_en=3.5):
    temp_YUV = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)
    res_RGB  = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)
    timg = img.astype(np.float32)
    for i in tqdm(range(timg.shape[0])):
        for j in range(timg.shape[1]):
            ##############################################################
            # Note that, should be careful about the RGB or BGR order
            # Hint: check the transformation matrix to convert RGB to YUV
            ##############################################################
            ## write your code here
            Y = 0.299 * timg[i, j, 0] + 0.587 * timg[i, j, 1] + 0.114 * timg[i, j, 2]
            U = -0.147 * timg[i, j, 0] - 0.289 * timg[i, j, 1] + 0.436 * timg[i, j, 2]
            V = 0.615 * timg[i, j, 0] - 0.515 * timg[i, j, 1] - 0.100 * timg[i, j, 2]

        ## 1. save temp_YUV for visualization
            temp_YUV[i, j, 0] = Y
            temp_YUV[i, j, 1] = U
            temp_YUV[i, j, 2] = V

            ## 2. enhance Y and convert YUV back to the RGB
            Y_enhanced = Y * lightness_en
            R_enhanced = max(0, min(255, Y_enhanced + 1.14 * V))
            G_enhanced = max(0, min(255, Y_enhanced - 0.39 * U - 0.58 * V))
            B_enhanced = max(0, min(255, Y_enhanced + 2.03 * U))

            ## 3. store the enhanced RGB
            res_RGB[i, j, 0] = R_enhanced
            res_RGB[i, j, 1] = G_enhanced
            res_RGB[i, j, 2] = B_enhanced

            #############################################################
            # end of your code
            #############################################################
            # pass
            #############################################################
            # (Optional) consider more efficent way to implement such a conversion
            #############################################################
    return temp_YUV, res_RGB

if __name__ == '__main__':
    img = cv2.imread("test/Lena.jpg")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # imgyuv, res_rgb = RGB2YUV_enhance(img)
    # imgcmy = convert(img, 'RGB2CMY')
    # resimg = convert(imgcmy, 'CMY2RGB')
    # imgyiq = convert(img, 'RGB2YIQ')
    # resimg = convert(imgyiq, 'YIQ2RGB')
    imgyuv = convert(img, 'RGB2YUV')
    resimg = convert(imgyuv, 'YUV2RGB')
    plt.figure()
    plt.suptitle('color space conversion Result')
    plt.subplot(221)
    plt.imshow(resimg[:,:,0], cmap='gray')
    plt.subplot(222)
    plt.imshow(resimg[:,:,1], cmap='gray')
    plt.subplot(223)
    plt.imshow(resimg[:,:,2], cmap='gray')
    plt.subplot(224)
    plt.imshow(resimg)
    plt.show()