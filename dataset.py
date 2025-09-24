import os, cv2, numpy as np, pandas as pd, matplotlib.pyplot as plt

kernel_sharpen = np.array([[-1, -1, -1],
                    [-1, 8, -1],
                    [-1, -1, -1]])*-1

def multiplication(kernel, submatrix):
    kernel_size = len(kernel)
    sum = 0
    for i in range(kernel_size):
        for j in range(kernel_size):
            sum += kernel[i][j]*submatrix[i][j]
    return sum

def convolution(kernel, matrix):
    kernel_size = len(kernel)
    padded = np.zeros((len(matrix)+2, len(matrix[0])+2))
    ret = np.zeros_like(padded)
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            padded[i+1][j+1] = matrix[i][j]

    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            minplus = kernel_size//2
            start_y = i+1-minplus
            start_x = j+1-minplus
            ret[i+1][j+1] = multiplication(
                kernel, 
                [arr[start_x:start_x+kernel_size] for arr in padded[start_y:start_y+kernel_size]]
                )
    return np.round(ret)
            

def get_dataset():
    path_normal = "TB_Chest_Radiography_Database/Normal"
    path_tb = "TB_Chest_Radiography_Database/Tuberculosis"
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

        x_tb.append(image)

    return x_normal, x_tb

def get_dataset_test():
    path_normal = "TBX11K/imgs/health"
    path_tb = "TBX11K/imgs/tb"
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

        x_tb.append(image)

    return x_normal, x_tb
    