import numpy as np
from scipy import linalg

yiq_from_rgb = np.array([[0.299     ,  0.587     ,  0.114     ],
                         [0.59590059, -0.27455667, -0.32134392],
                         [0.21153661, -0.52273617,  0.31119955]])

rgb_from_yiq = linalg.inv(yiq_from_rgb)

yuv_from_rgb = np.array([[ 0.299     ,  0.587     ,  0.114      ],
                         [-0.14714119, -0.28886916,  0.43601035 ],
                         [ 0.61497538, -0.51496512, -0.10001026 ]])

rgb_from_yuv = linalg.inv(yuv_from_rgb)

def convert(img, conv_type):
    if conv_type in ['YIQ2RGB', 'RGB2YIQ', 'RGB2YUV', 'YUV2RGB', 'CMY2RGB', 'RGB2CMY']:
        if conv_type == 'RGB2YIQ':
            return _convert(yiq_from_rgb, img)
        if conv_type ==  'YIQ2RGB':
            return _convert(rgb_from_yiq, img)
        if conv_type == 'RGB2YUV':
            return _convert(yuv_from_rgb, img)
        if conv_type == 'YUV2RGB':
            return _convert(rgb_from_yuv, img)
        if conv_type == 'CMY2RGB' or conv_type == 'RGB2CMY':
            return 1-img
    else:
        raise Exception('convert_type error! you should input:', ['YIQ2RGB', 'RGB2YIQ', 'RGB2YUV', 'YUV2RGB', 'CMY2RGB', 'RGB2CMY']) 

def _convert(matrix, arr):
    return (arr.astype(matrix.dtype) @ matrix.T).astype(np.int8)