import os, cv2, numpy as np, pandas as pd, matplotlib.pyplot as plt
from segmentation import get_segmentation
from joblib import load

MODELPATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),"model","svm.joblib")

class ImageProcessor:
    def __init__(self):
        self.clf = load(MODELPATH)

    def process(self,img):
        segmented = get_segmentation(img)
        result = self.clf.predict(segmented.flatten().reshape(1,-1))
        return result

