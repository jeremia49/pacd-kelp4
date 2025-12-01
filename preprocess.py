import os, cv2, numpy as np, pandas as pd, matplotlib.pyplot as plt

kernel_sharpen = np.array([[-1, -1, -1],
                    [-1, 8, -1],
                    [-1, -1, -1]])

kernel_dilation = np.ones((5, 5), np.uint8)

def preprocess(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Log transform
    image = (image)/255
    c = 255 / (1+np.max(image))
    image = c*np.log(1+image)
    image = np.array(image, dtype=np.uint8)

    image = cv2.equalizeHist(image)

    # image = cv2.equalizeHist(image)
    image = cv2.resize(image, (128, 128))
    # image = convolution(kernel_sharpen, image)
    image = abs(image+cv2.filter2D(image,-1,kernel_sharpen))
    image = cv2.bitwise_not(image)
    image = cv2.dilate(image, kernel_dilation, iterations=2)
    image = cv2.erode(image, kernel_dilation, iterations=2)
    image = cv2.copyMakeBorder(image, 2,2, 2, 2, borderType=cv2.BORDER_CONSTANT)
    return image