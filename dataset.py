import os, cv2, numpy as np, pandas as pd, matplotlib.pyplot as plt

kernel_sharpen = np.array([[-1, -1, -1],
                    [-1, 8, -1],
                    [-1, -1, -1]])

kernel_dilation = np.ones((5, 5), np.uint8)



def get_dataset(path_normal = "TBX11K/imgs/combined_normal", path_tb = "TBX11K/imgs/combined_tb"):
    x_normal=[]
    x_tb=[]
    for i, img in enumerate(os.listdir(path_normal)):        
        image = cv2.imread(os.path.join(path_normal, img))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Log transform
        image = (image)/255
        c = 255 / (1+np.max(image))
        image = c*np.log(1+image)
        image = np.array(image, dtype=np.uint8)

        image = cv2.equalizeHist(image)

        # image = cv2.equalizeHist(image)
        image = cv2.resize(image, (512, 512))
        # image = convolution(kernel_sharpen, image)
        image = abs(image+cv2.filter2D(image,-1,kernel_sharpen))
        image = cv2.bitwise_not(image)
        image = cv2.dilate(image, kernel_dilation, iterations=2)
        image = cv2.erode(image, kernel_dilation, iterations=2)
        image = cv2.copyMakeBorder(image, 2,2, 2, 2, borderType=cv2.BORDER_CONSTANT)
        x_normal.append(image)
    
    for i, img in enumerate(os.listdir(path_tb)):
        image = cv2.imread(os.path.join(path_tb, img))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Log transform
        image = (image)/255
        c = 255 / (1+np.max(image))
        image = c*np.log(1+image)
        image = np.array(image, dtype=np.uint8)

        image = cv2.equalizeHist(image)


        # image = cv2.equalizeHist(image)
        image = cv2.resize(image, (512, 512))
        if image.shape != (512, 512):
            continue
        # image = convolution(kernel_sharpen, image)
        image = abs(image+cv2.filter2D(image,-1,kernel_sharpen))
        image = cv2.bitwise_not(image)
        image = cv2.dilate(image, kernel_dilation, iterations=2)
        image = cv2.erode(image, kernel_dilation, iterations=2)
        image = cv2.copyMakeBorder(image, 2,2, 2, 2, borderType=cv2.BORDER_CONSTANT)

        x_tb.append(image)

    return x_normal, x_tb

def get_dataset_test():
    # path_normal = "Chest-X-Ray/normal"
    # path_tb = "Chest-X-Ray/tb"
    path_normal = "TB_Chest_Radiography_Database/Normal"
    path_tb = "TB_Chest_Radiography_Database/Tuberculosis"
    return get_dataset(path_normal, path_tb)