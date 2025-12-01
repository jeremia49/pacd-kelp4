import os, cv2, numpy as np, pandas as pd, matplotlib.pyplot as plt

def preprocess(image):
    image = (image)/255
    c = 255 / (1+np.max(image))
    image = c*np.log(1+image)
    image = np.array(image, dtype=np.uint8)

    ycrcb_img = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    ycrcb_img[:, :, 0] = cv2.equalizeHist(ycrcb_img[:, :, 0])
    image = cv2.cvtColor(ycrcb_img, cv2.COLOR_YCrCb2BGR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    image = cv2.resize(image, (512, 512))
    return image