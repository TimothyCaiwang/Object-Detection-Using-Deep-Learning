**object-detection-using-deep-learning**
Python script for object detection in small image patches and stitching into an entire orthomosaic image


Due to the GPU computational limitations, deep learning models cannot handle the large-scene orthomosaic image covering the whole study area. Therefore, a sliding window technique was used to run the image to crop image patches of 512×512-pixels with a stride value of 256 in the horizontal and vertical directions. 

The optimal model during the training process was applied to each image patch to extract boundaries of each strawberry plant. The detection results of image patches are then stitched to generate a shp format boundary file based on the algorthim proposed by Carvalho et al. (2020).


Carvalho, O. L. F. D., de Carvalho Júnior, O. A., Albuquerque, A. O. D., Bem, P. P. D., Silva, C. R., Ferreira, P. H. G., ... & Borges, D. L. (2020). Instance segmentation for large, multi-channel remote sensing imagery using mask-RCNN and a mosaicking approach. Remote Sensing, 13(1), 39.
