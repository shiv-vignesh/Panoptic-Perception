

import cv2
import numpy as np
import os
import torch

LANE_VIS_COLORS = [
    (255, 255, 255),  # single white
    (0, 255, 255),    # single yellow
    (0, 200, 255),    # double yellow
    (255, 128, 0),    # road curb
]

def visualize_batch_detection(img_draw:torch.Tensor, 
                              targets:torch.Tensor,
                              H:int, W:int,
                              batch_index=0):    

    # collect targets for this specific image
    t = targets[targets[:,0] == batch_index]

    # convert xywh normalized -> pixel PascalVOC (x1,y1,x2,y2)
    boxes = []
    classes = []
    for row in t:
        _, cls, xc, yc, w, h = row

        xc *= W
        yc *= H
        w *= W
        h *= H

        x1 = xc - w/2
        y1 = yc - h/2
        x2 = xc + w/2
        y2 = yc + h/2

        boxes.append([x1, y1, x2, y2])
        classes.append(int(cls))

    # draw bounding boxes on the image
    
    for (x1,y1,x2,y2), cls in zip(boxes, classes):
        cv2.rectangle(img_draw,
                      (int(x1), int(y1)),
                      (int(x2), int(y2)),
                      (0,255,0), 2)
        cv2.putText(img_draw, str(cls),
                    (int(x1), int(y1)-5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0,255,0), 2)

def visualize_batch_lane(img_draw:torch.Tensor, 
                        lane_targets:torch.Tensor,
                        H:int, W:int,
                        batch_index:int,
                        lane_categories=None):

    lt = lane_targets[batch_index]  # (max_lanes, 78)
    lc = lane_categories[batch_index] if lane_categories is not None else None
    n_offsets = lt.shape[1] - 6  # 72

    # y-positions: 1.0 (bottom) to 0.0 (top), matching CLRNet prior_ys
    prior_ys = np.linspace(1.0, 0.0, n_offsets)

    for i in range(lt.shape[0]):
        if lt[i, 0].item() < 0.5:  # not valid
            continue

        xs = lt[i, 6:].cpu().numpy()  # (72,) normalized x-coords
        cat_idx = int(lc[i].item()) if lc is not None and lc[i].item() >= 0 else 0
        color = LANE_VIS_COLORS[cat_idx % len(LANE_VIS_COLORS)]

        pts = []
        for j in range(n_offsets):
            if xs[j] > -1e4:
                x_pix = int(xs[j] * (W - 1))
                y_pix = int(prior_ys[j] * (H - 1))
                pts.append((x_pix, y_pix))

        for k in range(len(pts) - 1):
            cv2.line(img_draw, pts[k], pts[k + 1], color, 2)
    
def visualize_batch_segmentation(seg:torch.Tensor, batch_index:int=0):

    if seg[batch_index].ndim == 3:
        seg_vis = seg[batch_index].permute(1, 2, 0).cpu().numpy()
    else:
        seg_vis = seg[batch_index].cpu().numpy()

    # Scale segmentation class IDs for visibility (multiply by factor)
    # BDD100K has ~19 semantic classes, scale to spread across 0-255
    seg_scaled = (seg_vis * 12).clip(0, 255).astype(np.uint8)

    return seg_scaled

def visualize_batch_drivable(drivable:torch.Tensor, batch_index:int=0):

    if drivable[batch_index].ndim == 3:
        drivable_vis = drivable[batch_index].permute(1, 2, 0).cpu().numpy()
    else:
        drivable_vis = drivable[batch_index].cpu().numpy()

    # BDD100K drivable area classes: 0=background, 1=direct, 2=alternative
    # Create color visualization for proper visibility
    drivable_colored = np.zeros((*drivable_vis.shape, 3), dtype=np.uint8)
    drivable_colored[drivable_vis == 1] = [0, 255, 0]    # Direct drivable: green
    drivable_colored[drivable_vis == 2] = [0, 255, 255]  # Alternative: yellow

    return drivable_colored

def visualize_batch(images, seg, drivable, targets, save_dir, batch_index=0,
                    lane_targets=None, lane_categories=None):
    """
    images: tensor (B,3,H,W) normalized to [0,1]
    targets: tensor (N,6) -> (batch_idx, class, x_center, y_center, w, h)
    lane_targets: tensor (B, max_lanes, 78) or None
    lane_categories: tensor (B, max_lanes) or None
    batch_index: index of image in batch to visualize
    """

    img = images[batch_index].permute(1,2,0).cpu().numpy()
    img = (img * 255).astype(np.uint8)
    img_draw = img.copy()

    H, W, _ = img.shape

    visualize_batch_detection(img_draw, targets, 
                            H=H, W=W, batch_index=batch_index)

    # draw lane targets
    if lane_targets is not None:
        visualize_batch_lane(img_draw, lane_targets,
                            H=H, W=W, batch_index=batch_index,
                            lane_categories=lane_categories)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    cv2.imwrite(f'{save_dir}/sample_batch_{batch_index}.png', cv2.cvtColor(img_draw, cv2.COLOR_RGB2BGR))

    if seg is not None:
        seg_scaled = visualize_batch_segmentation(seg, batch_index=0)
        cv2.imwrite(f'{save_dir}/sample_batch_seg_{batch_index}.png', cv2.cvtColor(seg_scaled, cv2.COLOR_RGB2BGR))

    if drivable is not None:
        drivable_colored = visualize_batch_drivable(drivable, batch_index)
        cv2.imwrite(f'{save_dir}/sample_batch_drivable_{batch_index}.png', drivable_colored)