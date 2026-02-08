"""
Mosaic augmentation for YOLOP-style training.
Combines 4 images to create diverse scenes with varied scales.
Particularly effective for small objects and long-tail classes.
"""

import cv2
import numpy as np
import random
import albumentations as A

def mosaic_augmentation(images_list, bboxes_list, class_labels_list, 
                    segs_list, drivables_list, output_size=(640, 640)):                
    """                                                                                                                                 
    Create mosaic augmentation by combining 4 images.                                                                                   
                                                                                                                                        
    Args:                                                                                                                               
        images_list: list of 4 images (each HWC, BGR)                                                                                   
        bboxes_list: list of 4 bbox arrays, each Nx4 in pascal_voc format [x1, y1, x2, y2] original pixel coords                        
        class_labels_list: list of 4 class label arrays                                                                                 
        segs_list: list of 4 segmentation masks or None                                                                                 
        drivables_list: list of 4 drivable masks or None                                                                                
        output_size: output image size (height, width)                                                                                  

    Returns:                                                                                                                            
        mosaic_img, mosaic_labels (Nx5 normalized xywh), mosaic_seg, mosaic_drivable                                                    
    """                                                                                                                                 
    assert len(images_list) == 4, "Mosaic requires exactly 4 images"                                                                    
                                                                                                                                        
    h_out, w_out = output_size                                                                                                          
                                                                                                                                        
    # Create output arrays                                                                                                              
    mosaic_img = np.full((h_out, w_out, 3), 114, dtype=np.uint8)                                                                        
    mosaic_labels = []                                                                                                                  
                                                                                                                                        
    # Initialize masks if provided                                                                                                      
    has_seg = segs_list[0] is not None                                                                                                  
    has_drivable = drivables_list[0] is not None                                                                                        
                                                                                                                                        
    mosaic_seg = np.zeros((h_out, w_out), dtype=np.uint8) if has_seg else None                                                          
    mosaic_drivable = np.zeros((h_out, w_out), dtype=np.uint8) if has_drivable else None                                                
                                                                                                                                        
    # Random center point for dividing the mosaic                                                                                       
    yc = int(random.uniform(0.4 * h_out, 0.6 * h_out))                                                                                  
    xc = int(random.uniform(0.4 * w_out, 0.6 * w_out))                                                                                  
                                                                                                                                        
    # Define placement regions for 4 images                                                                                             
    placements = [                                                                                                                      
        (0, 0, xc, yc),           # top-left                                                                                            
        (xc, 0, w_out, yc),       # top-right                                                                                           
        (0, yc, xc, h_out),       # bottom-left                                                                                         
        (xc, yc, w_out, h_out)    # bottom-right                                                                                        
    ]                                                                                                                                   
                                                                                                                                        
    for idx, (img, bboxes, class_labels, seg, drivable) in enumerate(                                                                   
        zip(images_list, bboxes_list, class_labels_list, segs_list, drivables_list)                                                     
    ):                                                                                                                                  
        h, w = img.shape[:2]                                                                                                            
                                                                                                                                        
        # Get placement region                                                                                                          
        x1_place, y1_place, x2_place, y2_place = placements[idx]                                                                        
        place_h = y2_place - y1_place                                                                                                   
        place_w = x2_place - x1_place                                                                                                   
                                                                                                                                        
        # Resize to fit placement region                                                                                                
        scale = min(place_w / w, place_h / h)                                                                                           
        new_w = int(w * scale)                                                                                                          
        new_h = int(h * scale)                                                                                                          
                                                                                                                                        
        img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)                                                   
                                                                                                                                        
        if has_seg and seg is not None:                                                                                                 
            seg_resized = cv2.resize(seg, (new_w, new_h), interpolation=cv2.INTER_NEAREST)                                              
        if has_drivable and drivable is not None:                                                                                       
            drivable_resized = cv2.resize(drivable, (new_w, new_h), interpolation=cv2.INTER_NEAREST)                                    
                                                                                                                                        
        # Calculate actual placement coordinates                                                                                        
        x1_actual, y1_actual = x1_place, y1_place                                                                                       
        x2_actual = min(x1_actual + new_w, w_out)                                                                                       
        y2_actual = min(y1_actual + new_h, h_out)                                                                                       
                                                                                                                                        
        # Place image                                                                                                                   
        mosaic_img[y1_actual:y2_actual, x1_actual:x2_actual] = img_resized[:y2_actual - y1_actual, :x2_actual - x1_actual]              
                                                                                                                                        
        # Place masks                                                                                                                   
        if has_seg and seg is not None:                                                                                                 
            mosaic_seg[y1_actual:y2_actual, x1_actual:x2_actual] = seg_resized[:y2_actual - y1_actual, :x2_actual - x1_actual]          
        if has_drivable and drivable is not None:                                                                                       
            mosaic_drivable[y1_actual:y2_actual, x1_actual:x2_actual] = drivable_resized[:y2_actual - y1_actual, :x2_actual - x1_actual]
                                                                                                                                        
        # Transform bboxes                                                                                                              
        if len(bboxes) > 0:                                                                                                             
            for bbox, cls in zip(bboxes, class_labels):                                                                                 
                x1, y1, x2, y2 = bbox                                                                                                   
                                                                                                                                        
                # Scale to resized tile                                                                                                 
                x1_scaled = x1 * scale                                                                                                  
                y1_scaled = y1 * scale                                                                                                  
                x2_scaled = x2 * scale                                                                                                  
                y2_scaled = y2 * scale                                                                                                  
                                                                                                                                        
                # Offset to mosaic position                                                                                             
                x1_out = x1_scaled + x1_place                                                                                           
                y1_out = y1_scaled + y1_place                                                                                           
                x2_out = x2_scaled + x1_place                                                                                           
                y2_out = y2_scaled + y1_place                                                                                           
                                                                                                                                        
                # Clip to output bounds                                                                                                 
                x1_out = max(0, min(x1_out, w_out))                                                                                     
                y1_out = max(0, min(y1_out, h_out))                                                                                     
                x2_out = max(0, min(x2_out, w_out))                                                                                     
                y2_out = max(0, min(y2_out, h_out))                                                                                     
                                                                                                                                        
                # Convert to normalized xywh                                                                                            
                cx = (x1_out + x2_out) / 2 / w_out                                                                                      
                cy = (y1_out + y2_out) / 2 / h_out                                                                                      
                bw = (x2_out - x1_out) / w_out                                                                                          
                bh = (y2_out - y1_out) / h_out                                                                                          
                                                                                                                                        
                if 0 <= cx <= 1 and 0 <= cy <= 1 and bw > 0.001 and bh > 0.001:                                                         
                    mosaic_labels.append([cls, cx, cy, bw, bh])                                                                         
                                                                                                                                        
    mosaic_labels = np.array(mosaic_labels) if mosaic_labels else np.zeros((0, 5))                                                      

    return mosaic_img, mosaic_labels, mosaic_seg, mosaic_drivable 
