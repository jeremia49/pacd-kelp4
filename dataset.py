import os, cv2, numpy as np, pandas as pd, matplotlib.pyplot as plt

def get_dataset():
    path_normal = "TB_Chest_Radiography_Database/Normal"
    path_tb = "TB_Chest_Radiography_Database/Tuberculosis"
    x_normal=[]
    x_tb=[]
    for i, img in enumerate(os.listdir(path_normal)):        
        # Handle cropping
        image = cv2.imread(os.path.join(path_normal, img))
        gs = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gs,100,200)
        x, y, w, h = cv2.boundingRect(edges)
        temp = None
        try:
            temp = image[y:y+h, x:x+w, :].copy()
        except:
            pass

        if len(temp)>0:
            image = temp

        # Handle cropping to even smaller
        L2_image = cv2.imread(os.path.join(path_normal, img))
        gs = cv2.cvtColor(L2_image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gs,100,200,L2gradient=True)
        x, y, w, h = cv2.boundingRect(edges)
        temp = None
        try:
            temp = L2_image[y:y+h, x:x+w, :].copy()
        except:
            pass
        
        if len(temp)>0:
            image = temp

        image = (image)/255
        c = 255 / (1+np.max(image))
        image = c*np.log(1+image)
        image = np.array(image, dtype=np.uint8)

        ycrcb_img = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        ycrcb_img[:, :, 0] = cv2.equalizeHist(ycrcb_img[:, :, 0])
        image = cv2.cvtColor(ycrcb_img, cv2.COLOR_YCrCb2BGR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # image = cv2.equalizeHist(image)
        image = cv2.resize(image, (512, 512))

        x_normal.append(image)
    
    for i, img in enumerate(os.listdir(path_tb)):

        # Handle cropping
        image = cv2.imread(os.path.join(path_tb, img))
        gs = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gs,100,200)
        x, y, w, h = cv2.boundingRect(edges)
        temp = None
        try:
            temp = image[y:y+h, x:x+w, :].copy()
        except:
            pass

        if len(temp)>0:
            image = temp

        # Handle cropping to even smaller
        L2_image = cv2.imread(os.path.join(path_tb, img))
        gs = cv2.cvtColor(L2_image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gs,100,200,L2gradient=True)
        x, y, w, h = cv2.boundingRect(edges)
        temp = None
        try:
            temp = L2_image[y:y+h, x:x+w, :].copy()
        except:
            pass
        
        if len(temp)>0:
            image = temp

        # Log transform
        image = (image)/255
        c = 255 / (1+np.max(image))
        image = c*np.log(1+image)
        image = np.array(image, dtype=np.uint8)

        ycrcb_img = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        ycrcb_img[:, :, 0] = cv2.equalizeHist(ycrcb_img[:, :, 0])
        image = cv2.cvtColor(ycrcb_img, cv2.COLOR_YCrCb2BGR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


        # image = cv2.equalizeHist(image)
        image = cv2.resize(image, (512, 512))
        if image.shape != (512, 512):
            continue
        x_tb.append(image)

    return x_normal, x_tb
    
# def get_dataset_test():
#     path_normal = "TBX11K/imgs/health"
#     path_tb = "TBX11K/imgs/tb"
#     x_normal=[]
#     x_tb=[]
#     for img in os.listdir(path_normal):
#         image = cv2.imread(os.path.join(path_normal, img))
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#         x_normal.append(image)
    
#     for img in os.listdir(path_tb):
#         image = cv2.imread(os.path.join(path_tb, img))
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#         if image.shape != (512, 512):
#             continue
#         x_tb.append(image)

#     return x_normal, x_tb

def get_dataset_test():
    path_normal = "TBX11K/imgs/health"
    path_tb = "TBX11K/imgs/tb"
    x_normal=[]
    x_tb=[]
    for i, img in enumerate(os.listdir(path_normal)):        
        # Handle cropping
        image = cv2.imread(os.path.join(path_normal, img))
        gs = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gs,100,200)
        x, y, w, h = cv2.boundingRect(edges)
        temp = None
        try:
            temp = image[y:y+h, x:x+w, :].copy()
        except:
            pass

        if len(temp)>0:
            image = temp

        # Handle cropping to even smaller
        L2_image = cv2.imread(os.path.join(path_normal, img))
        gs = cv2.cvtColor(L2_image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gs,100,200,L2gradient=True)
        x, y, w, h = cv2.boundingRect(edges)
        temp = None
        try:
            temp = L2_image[y:y+h, x:x+w, :].copy()
        except:
            pass
        
        if len(temp)>0:
            image = temp

        image = (image)/255
        c = 255 / (1+np.max(image))
        image = c*np.log(1+image)
        image = np.array(image, dtype=np.uint8)

        ycrcb_img = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        ycrcb_img[:, :, 0] = cv2.equalizeHist(ycrcb_img[:, :, 0])
        image = cv2.cvtColor(ycrcb_img, cv2.COLOR_YCrCb2BGR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # image = cv2.equalizeHist(image)
        image = cv2.resize(image, (512, 512))

        x_normal.append(image)
    
    for i, img in enumerate(os.listdir(path_tb)):

        # Handle cropping
        image = cv2.imread(os.path.join(path_tb, img))
        gs = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gs,100,200)
        x, y, w, h = cv2.boundingRect(edges)
        temp = None
        try:
            temp = image[y:y+h, x:x+w, :].copy()
        except:
            pass

        if len(temp)>0:
            image = temp

        # Handle cropping to even smaller
        L2_image = cv2.imread(os.path.join(path_tb, img))
        gs = cv2.cvtColor(L2_image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gs,100,200,L2gradient=True)
        x, y, w, h = cv2.boundingRect(edges)
        temp = None
        try:
            temp = L2_image[y:y+h, x:x+w, :].copy()
        except:
            pass
        
        if len(temp)>0:
            image = temp

        # Log transform
        image = (image)/255
        c = 255 / (1+np.max(image))
        image = c*np.log(1+image)
        image = np.array(image, dtype=np.uint8)

        ycrcb_img = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        ycrcb_img[:, :, 0] = cv2.equalizeHist(ycrcb_img[:, :, 0])
        image = cv2.cvtColor(ycrcb_img, cv2.COLOR_YCrCb2BGR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


        # image = cv2.equalizeHist(image)
        image = cv2.resize(image, (512, 512))
        if image.shape != (512, 512):
            continue
        x_tb.append(image)

    return x_normal, x_tb
    