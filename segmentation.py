import cv2, numpy as np

    
def get_segmentation(source): 
    ret3,th3 = cv2.threshold(source,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    return th3
