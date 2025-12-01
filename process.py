import os, cv2, numpy as np, pandas as pd, matplotlib.pyplot as plt
from segmentation import get_segmentation
from joblib import load

MODELPATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),"model","svc.joblib")
LABELENCODERPATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),"model","le.joblib")

class ImageProcessor:
    def __init__(self):
        self.clf = load(MODELPATH)
        self.le = load(LABELENCODERPATH)

    def process(self,img):
        segmented = get_segmentation(img)
        result = self.clf.predict(segmented.flatten().reshape(1,-1))
        label = self.decode(result)
        return label[0]
    
    def decode(self, result):
        return self.le.inverse_transform(result)

