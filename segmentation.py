import cv2, numpy as np
from collections import deque
def bfs(matrix, y, x, dist = 0, visited=None):

    q=deque()
    q.append((y,x,dist))
    SIZE = 0
    while len(q)!=0:
        y, x, dist= q.popleft()
        if(y>=len(matrix) or x >=len(matrix[0])):
            continue
        if(y<0 or x<0):
            continue
        if(matrix[y][x] != 255):
            continue
        SIZE+=1
        delta_y = [0, 1, 0, -1]
        delta_x = [1, 0, -1, 0]

        for i, _ in enumerate(delta_y):
            new_y = y+delta_y[i]
            new_x = x+delta_x[i]
            if(new_y>=len(matrix) or new_x >=len(matrix[0])):
                continue
            if(new_y<0 or new_x<0):
                continue
            if visited[new_y][new_x] == False:
                visited[new_y][new_x]=True
                q.append((new_y,new_x,dist+1))
    return SIZE
def fill(matrix, y, x, fill=0, dist = 0, visited=None):

    q=deque()
    q.append((y,x,dist))
    SIZE = 0
    while len(q)!=0:
        y, x, dist= q.popleft()
        if(y>=len(matrix) or x >=len(matrix[0])):
            continue
        if(y<0 or x<0):
            continue
        if(matrix[y][x] == 0):
            continue
        matrix[y][x] = fill
        delta_y = [0, 1, 0, -1]
        delta_x = [1, 0, -1, 0]

        for i, _ in enumerate(delta_y):
            new_y = y+delta_y[i]
            new_x = x+delta_x[i]
            if(new_y>=len(matrix) or new_x >=len(matrix[0])):
                continue
            if(new_y<0 or new_x<0):
                continue
            if visited[new_y][new_x] == False:
                visited[new_y][new_x]=True
                q.append((new_y,new_x,dist+1))
    return matrix
def fill_white_areas(image):
    white_areas=[]
    visited = [[False]*len(image[0]) for i in range(len(image))]

    for i, item_i in enumerate(image):
        for j, item_j in enumerate(item_i):
            if item_j==255 and not visited[i][j]:
                white_areas.append((bfs(image,i,j, visited=visited), (i,j)))
    white_areas = sorted(white_areas)
    visited = [[False]*len(image[0]) for i in range(len(image))]
    
    for a,b in white_areas[:-2]:
        y, x = b
        image = fill(image, y, x, visited=visited, fill=0)
    return image
def get_segmentation(source): 

    # ret3,th3 = cv2.threshold(source,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    ret3,th3 = cv2.threshold(source,0,255,cv2.THRESH_OTSU)

    contours, hierarchy = cv2.findContours(image=th3, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)
                                      
    cv2.drawContours(image=th3, contours=contours, contourIdx=-1, color=(0, 255, 0), thickness=10, lineType=cv2.LINE_AA)
    # th3 = cv2.adaptiveThreshold(source,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
    #         cv2.THRESH_BINARY,11,2)
    # blur = cv2.GaussianBlur(source,(5,5),0)
    # ret3,th3 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    # th3=fill_white_areas(th3)
    
    return th3
