# polygon-yolov5
This repository is based on Ultralytics/yolov5, with adjustments to enable polygon prediction boxes.

## Description
The codes are based on Ultralytics/yolov5, and several functions are added and modified to enable polygon prediction boxes.

The modifications compared with Ultralytics/yolov5 and their brief descriptions are summarized below:

  1. data/polygon_ucas.yaml : Exemplar UCAS-AOD dataset to test the effects of polygon boxes
  2. data/images/UCAS-AOD : For the inference of polygon-yolov5s-ucas.pt

  3. models/common.py :
    <br/> 3.1. class Polygon_NMS : Non-Maximum Suppression (NMS) module for Polygon Boxes
    <br/> 3.2. class Polygon_AutoShape : Polygon Version of Original AutoShape, input-robust polygon model wrapper for passing cv2/np/PIL/torch inputs. Includes preprocessing, inference and Polygon_NMS
    <br/> 3.3. class Polygon_Detections : Polygon detections class for Polygon-YOLOv5 inference results
  4. models/polygon_yolov5s_ucas.yaml : Configuration file of polygon yolov5s for exemplar UCAS-AOD dataset
  5. **models/yolo.py :**
    <br/> 5.1. class Polygon_Detect : Detect head for polygon yolov5 models with polygon box prediction
    <br/> 5.2. class Polygon_Model : Polygon yolov5 models with polygon box prediction
    
  6. **utils/iou_cuda : CUDA extension for iou computation of polygon boxes**
    <br/> 6.1. extensions.cpp : CUDA extension file
    <br/> 6.2. inter_union_cuda.cu : CUDA code for computing iou of polygon boxes
    <br/> 6.3. setup.py : for building CUDA extensions module polygon_inter_union_cuda, with two functions polygon_inter_union_cuda and polygon_b_inter_union_cuda
  **7. utils/autoanchor.py :**
    <br/> 7.1. def polygon_check_anchors : Polygon version of original check_anchors
    <br/> 7.2. def polygon_kmean_anchors : Create kmeans-evolved anchors from polygon-enabled training dataset, use minimum outter bounding box as approximations
  **8. utils/datasets.py :**
    <br/> 8.1. def polygon_random_perspective : Data augmentation for datasets with polygon boxes (augmentation effects: HSV-Hue, HSV-Saturation, HSV-Value, rotation, translation, scale, shear, perspective, flip up-down, flip left-right, mosaic, mixup)
    <br/> 8.2. def polygon_box_candidates : Polygon version of original box_candidates
    <br/> 8.3. class Polygon_LoadImagesAndLabels : Polygon version of original LoadImagesAndLabels
    <br/> 8.4. def polygon_load_mosaic : Loads images in a 4-mosaic, with polygon boxes
    <br/> 8.5. def polygon_load_mosaic9 : Loads images in a 9-mosaic, with polygon boxes
    <br/> 8.6. def polygon_verify_image_label : Verify one image-label pair for polygon datasets
    <br/> 8.7. def create_dataloader : Has been modified to include polygon datasets
  **9. utils/general.py :**
    <br/> 9.1. def xyxyxyxyn2xyxyxyxy : Convert normalized xyxyxyxy or segments into pixel xyxyxyxy or segments
    <br/> 9.2. def polygon_segment2box : Convert 1 segment label to 1 polygon box label
    <br/> 9.3. def polygon_segments2boxes : Convert segment labels to polygon box labels
    <br/> 9.4. def polygon_scale_coords : Rescale polygon coords (xyxyxyxy) from img1_shape to img0_shape
    <br/> 9.5. def polygon_clip_coords : Clip bounding polygon xyxyxyxy bounding boxes to image shape (height, width)
    <br/> 9.6. def polygon_inter_union_cpu : iou computation (polygon) with cpu
    <br/> 9.7. def polygon_box_iou : Compute iou of polygon boxes via cpu or cuda
    <br/> 9.8. def polygon_b_inter_union_cpu : iou computation (polygon) with cpu for class Polygon_ComputeLoss in loss.py
    <br/> 9.9. def polygon_bbox_iou : Compute iou of polygon boxes for class Polygon_ComputeLoss in loss.py via cpu or cuda
    <br/> 9.10. def polygon_non_max_suppression : Runs Non-Maximum Suppression (NMS) on inference results for polygon boxes
    <br/> 9.11. def polygon_nms_kernel : Non maximum suppression kernel for polygon-enabled boxes
    <br/> 9.12. def order_corners : Return sorted corners for loss.py::class Polygon_ComputeLoss::build_targets
  **10. utils/loss.py :**
    <br/> 10.1. class Polygon_ComputeLoss : Compute loss for polygon boxes
  11. utils/metrics.py :
    <br/> 11.1. class Polygon_ConfusionMatrix : Polygon version of original ConfusionMatrix
  12. utils/plots.py :
    <br/> 12.1. def polygon_plot_one_box : Plot one polygon box on image
    <br/> 12.2. def polygon_plot_one_box_PIL : Plot one polygon box on image via PIL
    <br/> 12.3. def polygon_output_to_target : Convert model output to target format (batch_id, class_id, x1, y1, x2, y2, x3, y3, x4, y4, conf)
    <br/> 12.4. def polygon_plot_images : Polygon version of original plot_images
    <br/> 12.5. def polygon_plot_test_txt : Polygon version of original plot_test_txt
    <br/> 12.6. def polygon_plot_targets_txt : Polygon version of original plot_targets_txt
    <br/> 12.7. def polygon_plot_labels : Polygon version of original plot_labels
  
  **13. polygon_train.py : For training polygon-yolov5 models**
  **14. polygon_test.py : For testing polygon-yolov5 models**
  **15. polygon_detect.py : For detecting polygon-yolov5 models**
  16. requirements.py : Added python model shapely
  
## How Does Polygon Boxes Work? How Does Polygon Boxes Different from Axis-Aligned Boxes?
  1. build_targets in class Polygon_ComputeLoss & forward in class Polygon_Detect
<br/>
<img src="https://user-images.githubusercontent.com/87064748/124885337-bfaa4f00-e005-11eb-957c-404b3164ad7a.jpg" width="200">
  2. order_corners in general.py
<br/>
<img src="https://user-images.githubusercontent.com/87064748/124885357-c3d66c80-e005-11eb-90b3-d3335c2c37bf.jpg" width="200">
  3. Illustrations of box loss of polygon boxes
<br/>
<img src="https://user-images.githubusercontent.com/87064748/124885366-c5a03000-e005-11eb-974e-14d61956955f.jpg" width="100" height="200">

