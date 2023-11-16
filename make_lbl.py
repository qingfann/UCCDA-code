import numpy as np
import glob
import os
"""
data/home/Art/Exit_Sign/00014.jpg 18
data/home/Art/Exit_Sign/00005.jpg 18
data/home/Art/Exit_Sign/00012.jpg 18
data/home/Art/Exit_Sign/00004.jpg 18
data/home/Art/Exit_Sign/00016.jpg 18"""

path = "./data/Remote_Sensing/"


dir_names = ["NWPU_RESISC45"]


for dir_name in dir_names:
    classes = os.listdir(os.path.join(path,dir_name))
    print(os.path.join(path,dir_name+".txt"))
    f = open(os.path.join(path, dir_name+".txt"),"w",encoding='UTF-8')
    indices=[]
    pathes=[]
    for c_i,c_name in enumerate(classes):
        indices.append((c_i,c_name))
    indices = np.random.permutation(indices)
    for c_i,c_name in indices:
        c_pathes = glob.glob(path+dir_name+"/"+c_name+"/*.png")+glob.glob(path+dir_name+"/"+c_name+"/*.jpg")+glob.glob(path+dir_name+"/"+c_name+"/*.JPEG")+glob.glob(path+dir_name+"/"+c_name+"/*.tif")
        c_pathes = [c_path.replace(path,"data/Remote_Sensing/").replace("\\","/")+" "+str(c_i)+"\n" for c_path in c_pathes]
        c_pathes = np.random.permutation(c_pathes)
        for c_path in c_pathes:
            f.write(c_path)

    f.close()