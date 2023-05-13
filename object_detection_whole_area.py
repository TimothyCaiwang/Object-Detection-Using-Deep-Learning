# -*- coding: utf-8 -*-
"""
Created on Tue Jul 27 17:19:55 2021

@author: 13527
"""

# -*- coding: utf-8 -*-
import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt
import cv2
import time
from mrcnn.config import Config
from datetime import datetime

import gdal
import glob
#import shapefile


import geopandas
from shapely import geometry

import tensorflow as tf
import shutil

# Root directory of the project
# ROOT_DIR = os.getcwd()

ROOT_DIR = os.getcwd()
 
TEST_DIR = "/Orthos_Phenomics/"
Result_DIR = "/mask_result/"
Temp_DIR = "/Temp/"

# pic = "C:/Users/13527/Desktop/Research/Object Detection/Mask_RCNN-multi-bands/samples_drone/train_images/20201210-20-DEM-1cm-Clipped.tif'


# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
# Import COCO config
# sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))  # To find local version
# from samples.coco import coco
 
 
# Directory to save logs and trained model\canopy20210823T2340
#MODEL_DIR = os.path.join(ROOT_DIR, "logs","NIR-R-G_512","canopy20211020T0128")
MODEL_DIR = os.path.join(ROOT_DIR, "logs","canopy20220708T0104")
# canopy20210823T0102 512 size IRGB
# canopy20210813T1731 384 size IRGB
 
# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(MODEL_DIR ,"mask_rcnn_canopy_0080.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)
    print("cuiwei***********************")
 
# Directory of images to run detection on
# IMAGE_DIR = os.path.join(ROOT_DIR, "images")
 
class ShapesConfig(Config):
    """Configuration for training on the toy shapes dataset.
    Derives from the base Config class and overrides values specific
    to the toy shapes dataset.
    """
    # Give the configuration a recognizable name
    NAME = "shapes"
 
    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
 
    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # background + 3 shapes
 
    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 320
    IMAGE_MAX_DIM = 512
 
    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (16, 32, 64, 128, 256)  # anchor side in pixels
 
    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE =100
 
    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 300
 
    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 300
 
#import train_tongue
#class InferenceConfig(coco.CocoConfig):
class InferenceConfig(ShapesConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
 
    
def isIntersection(x1,y1,x2,y2,x3,y3,x4,y4):
    # xmin_a, xmax_a, ymin_a, ymax_a, xmin_b, xmax_b, ymin_b, ymax_b
    intersect_flag = True
    
    w1 = x2 - x1
    h1 = y2 - y1
    w2 = x4 - x3
    h2 = y4 - y3
    
    w = abs((x1+x2)/2 - (x3+x4)/2);
    h = abs((y1+y2)/2 - (y3+y4)/2);
    
    if w < (w1+w2)/2 and h < (h1+h2)/2:
        intersect_flag = True
    else:
        intersect_flag = False

    return intersect_flag



def IOU(box1, box2):
    y1, x1, y2, x2 = box1
    y3, x3, y4, x4 = box2
    iou = 0
    if_inter = isIntersection(x1,y1,x2,y2,x3,y3,x4,y4)
    #print(x1,y1,x2,y2,x3,y3,x4,y4)
    
    if if_inter == True:    
    	x_inter1 = max(x1, x3)
    	y_inter1 = max(y1, y3)
    	x_inter2 = min(x2, x4)
    	y_inter2 = min(y2, y4)
        
    	width_inter = abs(x_inter2 - x_inter1)
    	height_inter = abs(y_inter2 - y_inter1)
    	area_inter = width_inter * height_inter
        
    	width_box1 = abs(x2 - x1)
    	height_box1 = abs(y2 - y1)
        
    	width_box2 = abs(x4 - x3)
    	height_box2 = abs(y4 - y3)
        
    	area_box1 = width_box1 * height_box1
    	area_box2 = width_box2 * height_box2
    	area_union = area_box1 + area_box2 - area_inter
    	area_min = min(area_box1,area_box2)
    	iou = area_inter / area_min     
    else:
        iou = 0
    return iou
    
    
def mask_select(box_c, scores):
    del_if = np.zeros((len(scores),), dtype=np.uint8)
    w_c = abs(box_c[:,2]-box_c[:,0])
    h_c = abs(box_c[:,3]-box_c[:,1])
    area_c = w_c*h_c
    
    area_sp = area_c.argsort()
    area_sn = area_c[area_sp]
    
    for i in range(len(scores)-1):
        IOU_i = []
        for j in range(i+1,len(scores)):           
            IOU_ij = IOU(box_c[area_sp[i],:], box_c[area_sp[j],:])
            IOU_i.append(IOU_ij)
            #print(box_c[area_sp[i],:], box_c[area_sp[j],:])
            #print(IOU_ij)
        if np.max(IOU_i)>0.7 or scores[area_sp[i]]<0.8:
            del_if[area_sp[i]] = 1
        else:
            del_if[area_sp[i]] = 0
    return del_if
                
def equalizeHist_image(x):
    x = x.astype(np.float32)/65535*255
    x = x.astype(np.uint8)
    x = cv2.equalizeHist(x)
    return x
    
def del_file(filepath):
    """
    删除某一目录下的所有文件或文件夹
    :param filepath: 路径
    :return:
    """
    del_list = os.listdir(filepath)
    for f in del_list:
        file_path = os.path.join(filepath, f)
        if os.path.isfile(file_path):
            os.remove(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)
    
    
config = InferenceConfig()
 
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)
 
 
# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)
 
# Load weights trained on MS-COCO
model.load_weights(COCO_MODEL_PATH, by_name=True)
 
# COCO Class names
# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: class_names.index('teddy bear')
class_names = ['BG', 'Canopy']
# Load a random image from the images folder
# file_names = next(os.walk(TEST_DIR))[2]

#file_names = glob.glob(TEST_DIR + "*.tif")
file_names = os.listdir(TEST_DIR)



for f_i in file_names:
    box_c = np.empty(shape=(0,4))
    box_c = box_c.astype(np.uint8)
    
    scores = np.empty(shape=(0,))
    scores = scores.astype(np.float32)
    
    #masks = np.empty(shape=(512,512,0))
    #masks = masks.astype(np.uint8)
    
    pshift = np.empty(shape=(0,4))
    if(f_i.endswith('.tif')):
        image_path = os.path.join(TEST_DIR, f_i)
        img = gdal.Open(image_path)
        itr = img.GetGeoTransform()
        
        # out_shp = os.path.join(TEST_DIR, "20181218.shp")
    
        band_num = img.RasterCount # band number
        
        #img_f = img.astype(np.float32)/65535*255
        #img_u = img_f.astype(np.uint8)
        # img = img
        
        bandB = img.GetRasterBand(1).ReadAsArray()
        bandB = bandB.astype(np.float32)/65535*255
        bandB = bandB.astype(np.uint8)
        #del bandB
        
        bandG = img.GetRasterBand(2).ReadAsArray()
        bandG = bandG.astype(np.float32)/65535*255
        bandG = bandG.astype(np.uint8)
        #bandR = img_u.GetRasterBand(3).ReadAsArray()
        #bandNIR = img_u.GetRasterBand(4).ReadAsArray()
        bandRE = img.GetRasterBand(5).ReadAsArray()
        bandRE = bandRE.astype(np.float32)/65535*255
        bandRE = bandRE.astype(np.uint8)
        
        
        n = img.RasterXSize # 列数
        m = img.RasterYSize # 行数
        w = 256 # cropped image size 2w*2w
        s = 256 # sliding window size in the horiztonal and vertical directions
        del img
        
        tmasks = 0
        tttt = 0
        for j in range(w,m+s,s):
            for k in range(w,n+s,s):
                tttt = tttt+1
                print(tttt)
                x1 = j-w
                x2 = j+w
                y1 = k-w
                y2 = k+w
                if x2>m:
                    x2 = m
                    x1 = m-2*w
                if y2>n:
                    y2 = n
                    y1 = n-2*w
                
                # y = bandDEM1[(i-w):(i+w),(j-w):(j+w)]
                bandBx = bandB[x1:x2,y1:y2]
                bandGx = bandG[x1:x2,y1:y2]
                #bandRx = bandR[x1:x2,y1:y2]
                #bandNIRx = bandNIR[x1:x2,y1:y2]
                bandREx = bandRE[x1:x2,y1:y2]
                
                #bandBx_e = equalizeHist_image(bandBx)
                #bandGx_e = equalizeHist_image(bandGx)
                #bandRx_e = equalizeHist_image(bandRx)
                #bandNIRx_e = equalizeHist_image(bandNIRx)
                #bandREx_e = equalizeHist_image(bandREx)
                #bandNIRx = bandNIR[x1:x2,y1:y2]
                #bandREx = bandRE[x1:x2,y1:y2]
                image = np.array([bandREx,bandGx,bandBx])
                #image = image.astype(np.float32)/65535*255
                #image = image.astype(np.uint8)
                image = image.transpose(1,2,0)
                del bandBx
                del bandGx
                del bandREx
                
                results = model.detect([image], verbose=1)
                del image
                r = results[0]
                
                br = r['rois']
                #figsize=(16, 16)
                #_, ax1 = plt.subplots(1, figsize=figsize)
                #visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],class_names, r['scores'],ax = ax1, show_mask=True)
                
                #br[:,0] = br[:,0]+j-w
                #br[:,1] = br[:,1]+k-w
                #br[:,2] = br[:,2]+j-w
                #br[:,3] = br[:,3]+k-w
                
                # br: rois mapped to original orthomosaic image [m,n]
                if x2==m:
                    br[:,0] = br[:,0]+m-2*w
                    br[:,2] = br[:,2]+m-2*w
                else:
                    br[:,0] = br[:,0]+j-w
                    br[:,2] = br[:,2]+j-w
    
                if y2==n:
                    br[:,1] = br[:,1]+n-2*w
                    br[:,3] = br[:,3]+n-2*w
                else:
                    br[:,1] = br[:,1]+k-w
                    br[:,3] = br[:,3]+k-w
                
                # y1, x1, y2, x2 = box
                shift_jk = np.zeros([br.shape[0],4]) # record the offset in the original orthomosaic image;
                #shift_jk[:,0] = j-w
                #shift_jk[:,1] = k-w
                #shift_jk[:,2] = j-w
                #shift_jk[:,3] = k-w
                
                if x2==m:
                    shift_jk[:,0] = m-2*w
                    shift_jk[:,2] = m-2*w
                else:
                    shift_jk[:,0] = j-w
                    shift_jk[:,2] = j-w
    
                if y2==n:
                    shift_jk[:,1] = n-2*w
                    shift_jk[:,3] = n-2*w
                else:
                    shift_jk[:,1] = k-w
                    shift_jk[:,3] = k-w
                          
                
                box_c = np.append(box_c,br,axis=0)
                scores = np.append(scores,r['scores'],axis=0)
                r_mask = r['masks']
                r_mask.astype(np.uint8)
                del r
                
                r_shape = r_mask.shape
                lmask = r_shape[2]
                # t_masks = t_masks+lmask
                if lmask != 0:
                    for lmr in range(lmask):
                        r_maskt = r_mask[:,:,lmr]
                        r_maskn = str(lmr+tmasks)+'.npy'
                        r_maskp = os.path.join(Temp_DIR, r_maskn)
                        np.save(r_maskp,r_maskt)
                    tmasks = tmasks+lmask
                    #masks = np.append(masks,r_mask,axis=2)
                    pshift = np.append(pshift,shift_jk,axis=0)
                
                #del masks
        del bandB, bandG, bandRE
        del_if = mask_select(box_c, scores)
        # delete the small mask patches;
            
            #shape_type = 5
            #shp_w = shapefile.Writer(target=out_shp, shapeType=shape_type, autoBalance=1)
            
        xypa = []
        for t in range(len(scores)):
            if del_if[t] == 0:
                t_maskn = str(t)+'.npy'
                t_maskp = os.path.join(Temp_DIR,t_maskn)
                
                masks_i = np.load(t_maskp)
                masks_i = masks_i.astype(np.uint8)
                binary,contours,hierarchy = cv2.findContours(masks_i,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
                if len(contours) == 1:
                    contour_n = contours[0]
                else:
                    if len(contours[0]) < 3:
                        contour_n = contours[1]
                    else:
                        contour_n = contours[0]
                
                x = contour_n[:,0,0] + pshift[t,1] # column k-w
                y = contour_n[:,0,1] + pshift[t,0] # row j-w
                
                x = x.astype(np.float)
                y = y.astype(np.float)
                
                xp = itr[0]+itr[1]*x
                yp = itr[3]+itr[5]*y
                
                xyp = np.vstack((xp,yp))
                xyp = np.transpose(xyp)
                xlt = xyp.shape
                if xlt[0] >= 3:
                    xypa.append(geometry.Polygon(xyp))
        shp_si = f_i.find('.')
        shp_n = f_i[0:shp_si]+'.shp'
        if len(xypa)>0:
            shp_out = geopandas.GeoSeries(xypa,crs='EPSG:3857')
            shp_out.to_file(shp_n,driver='ESRI Shapefile')
            del xypa, xyp, xp, yp, binary, contours, hierarchy, shp_out
        del_file(Temp_DIR)
