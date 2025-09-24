import cv2, numpy as np


def get_segmentation(source):
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

    k = 2

    image = np.array(source,dtype=np.float32)
    pixel_values = np.array(image.reshape((-1, 3)) if len(image.shape) == 3 else image.reshape((-1, 1)))
    pixel_values = np.float32(pixel_values)

    retval, labels, centers = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_PP_CENTERS) 


    centers = np.uint8(centers) 

    segmented_data = centers[labels.flatten()] 

    segmented_image = segmented_data.reshape((image.shape)) 
    return segmented_image
