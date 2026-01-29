"""
Scribble Generation Module

This module provides multiple methods for generating training scribbles from 
segmentation masks. It simulates human annotation behavior by creating various
types of scribbles (boundary, curve, thinned) from error regions.

Point Sampling Strategies:

1. Boundary Scribbles (get_boundary_scribble):
   - Samples points along the boundary of error regions
   - Uses morphological gradient after erosion
   - Creates scribbles that trace object edges

2. Curve Scribbles (get_curve_scribble):
   - Randomly samples 3 control points from the error region
   - Fits quadratic Bezier curves through the points
   - Evaluates curves at 1024 uniformly spaced parameter values (t ∈ [0,1])
   - Generates 2-4 curves per region, selects longest ones
   - Creates natural, smooth scribbles through object interiors

3. Thinned Scribbles (get_thinned_scribble):
   - Uses Zhang-Suen thinning algorithm to skeletonize the region
   - Samples points along the skeleton
   - Creates scribbles following the region's medial axis

4. Robot Scribbles (TamedRobot):
   - Uses sophisticated graph-based point sampling (see tamed_robot.py)
   - Mimics DAVIS interactive annotation protocol
   - Samples points along longest paths in skeleton graphs

The main function get_scribble() randomly combines these methods to generate
diverse training data that helps the model generalize to different annotation styles.
"""

import numpy as np
import cv2
import bezier
from dataset.tamed_robot import TamedRobot
from dataset.mask_perturb import random_erode


def disk_kernel(size):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))

def get_boundary_scribble(region):
    # Draw along the boundary of an error region
    erode_size = np.random.randint(3, 50)
    eroded = cv2.erode(region, disk_kernel(erode_size))
    scribble = cv2.morphologyEx(eroded, cv2.MORPH_GRADIENT, np.ones((3,3)))

    h, w = region.shape
    for _ in range(4):
        lx, ly = np.random.randint(w), np.random.randint(h)
        lw, lh = np.random.randint(lx+1,w+1), np.random.randint(ly+1,h+1)
        scribble[ly:lh, lx:lw] = random_erode(scribble[ly:lh, lx:lw], min=5)

    return scribble

def get_curve_scribble(region, min_srb=2, max_srb=4, sort=True):
    """
    Generate curve-based scribbles by sampling points and fitting Bezier curves.
    
    This method creates smooth, natural-looking scribbles by:
    1. Randomly sampling 3 control points from the error region
    2. Fitting a quadratic Bezier curve through these points
    3. Evaluating the curve at 1024 uniformly spaced parameter values
    4. Rendering the curve as a polyline with thickness 3
    
    Point Sampling Details:
    - Control points are randomly selected from region pixels
    - Bezier curves use parametric evaluation: P(t) = Σ B_i(t) * P_i, t ∈ [0,1]
    - 1024 evaluation points provide smooth curves (can be adjusted)
    - Multiple curves are generated and the longest ones are kept
    
    Args:
        region: Binary mask of the error region to scribble
        min_srb: Minimum number of scribble lines to generate
        max_srb: Maximum number of scribble lines to generate
        sort: If True, generate more curves and select the longest ones
        
    Returns:
        Binary mask containing the curve scribbles
    """
    # Draw random curves
    num_lines = np.random.randint(min_srb, max_srb)

    scribbles = []
    lengths = []
    eval_pts = np.linspace(0.0, 1.0, 1024)
    if sort:
        # Generate more anyway, pick the best k at last
        num_gen = 10
    else:
        num_gen = num_lines
    for _ in range(num_gen):
        # Sample 3 random control points from the region
        region_indices = np.argwhere(region)
        include_idx = np.random.choice(region_indices.shape[0], size=3, replace=False)
        # Create Bezier curve nodes for x and y separately
        y_nodes = np.asfortranarray([
            [0.0, 0.5, 1.0],
            region_indices[include_idx, 0],
        ])
        x_nodes = np.asfortranarray([
            [0.0, 0.5, 1.0],
            region_indices[include_idx, 1],
        ])
        # Fit quadratic Bezier curves
        x_curve = bezier.Curve(x_nodes, degree=2)
        y_curve = bezier.Curve(y_nodes, degree=2)
        # Evaluate at 1024 uniformly spaced points (dense sampling for smoothness)
        x_pts = x_curve.evaluate_multi(eval_pts)
        y_pts = y_curve.evaluate_multi(eval_pts)

        this_scribble = np.zeros_like(region)
        pts = np.stack([x_pts[1,:], y_pts[1,:]], 1)
        pts = pts.reshape((-1, 1, 2)).astype(np.int32)
        # Draw polyline connecting all sampled points with thickness 3
        this_scribble = cv2.polylines(this_scribble, [pts], isClosed=False, color=(1), thickness=3)

        # Mask away path outside the allowed region, allow some error in labeling
        allowed_error = np.random.randint(3, 7)
        allowed_region = cv2.dilate(region, disk_kernel(allowed_error))
        this_scribble = this_scribble * allowed_region

        scribbles.append(this_scribble)
        lengths.append(this_scribble.sum())

    # Sort according to length, we want the long lines
    scribbles = [x for _, x in sorted(zip(lengths, scribbles), key=lambda pair: pair[0], reverse=True)]
    scribble = sum(scribbles[:num_lines])

    return (scribble>0.5).astype(np.uint8)

def get_thinned_scribble(region):
    # Use the thinning algorithm for scribbles
    thinned = (cv2.ximgproc.thinning(region*255, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)>128).astype(np.uint8)

    scribble = cv2.dilate(thinned, np.ones((3,3)))
    h, w = region.shape
    for _ in range(4):
        lx, ly = np.random.randint(w), np.random.randint(h)
        lw, lh = np.random.randint(lx+1,w+1), np.random.randint(ly+1,h+1)
        scribble[ly:lh, lx:lw] = random_erode(scribble[ly:lh, lx:lw], min=5)

    return scribble

robot = TamedRobot()
def get_scribble(mask, gt, from_zero):
    """
    Main scribble generation function with multiple sampling strategies.
    
    This function generates positive and negative scribbles for training by:
    1. Computing error regions (false positives and false negatives)
    2. Selecting a point sampling strategy (robot or manual methods)
    3. For each error region, randomly choosing scribble types
    4. Optionally combining multiple scribble types for diversity
    
    Point Sampling Strategies Used:
    - Robot (75% probability): Graph-based sampling via TamedRobot
    - Manual (25% probability): Random combination of boundary/thinned/curve scribbles
    
    Args:
        mask: Current segmentation prediction (0-255 grayscale)
        gt: Ground truth segmentation (0-255 grayscale)
        from_zero: If True, simulate annotation from scratch (no previous mask)
        
    Returns:
        tuple: (pos_scr, neg_scr) - Binary masks for positive and negative scribbles
    """
    mask = mask > 128
    gt = gt > 128

    if from_zero:
        use_robot = False
    else:
        if np.random.rand() < 0.75:
            use_robot = True
        else:
            use_robot = False

    # False positive and false negative
    fp = (mask & ~gt).astype(np.uint8)
    fn = (~mask & gt).astype(np.uint8)

    if use_robot:
        # The robot is similar to the DAVIS official one
        neg_scr = robot.interact(fp).astype(np.uint8)
        pos_scr = robot.interact(fn).astype(np.uint8)
        neg_scr = cv2.dilate(neg_scr, np.ones((3,3)))
        pos_scr = cv2.dilate(pos_scr, np.ones((3,3)))
        return pos_scr, neg_scr
    else:
        # Opening operator to remove noises
        opening_size = np.random.randint(5, 20)
        fp = cv2.morphologyEx(fp, cv2.MORPH_OPEN, disk_kernel(opening_size))
        fn = cv2.morphologyEx(fn, cv2.MORPH_OPEN, disk_kernel(opening_size))

        # Use connected error regions for processing
        scribbles = []
        for m in [fn, fp]:
            this_scribble = np.zeros_like(mask)
            num_labels, labels_im = cv2.connectedComponents(m)
            for n in range(1, num_labels):
                # Obtain scribble for this single region
                region_mask = (labels_im==n).astype(np.uint8)
                if region_mask.sum() < np.random.randint(10, 100):
                    continue

                # Initial pass, pick a scribble type
                pick = np.random.rand()
                if pick < 0.33:
                    region_scribble = get_boundary_scribble(region_mask)
                elif pick < 0.66:
                    region_scribble = get_thinned_scribble(region_mask)
                else:
                    region_scribble = get_curve_scribble(region_mask)
                this_scribble = (this_scribble | region_scribble)
                    
                # Optionally use a second scribble type
                pick = np.random.rand()
                if pick < 0.3:
                    pick = np.random.rand()
                    if pick < 0.33:
                        region_scribble = get_boundary_scribble(region_mask)
                    elif pick < 0.66:
                        region_scribble = get_thinned_scribble(region_mask)
                    else:
                        region_scribble = get_curve_scribble(region_mask)
                    this_scribble = (this_scribble | region_scribble)

            scribbles.append(this_scribble)

        # Sometimes we just draw scribbles referring only to the GT but not the given mask
        if np.random.rand() < 0.3 or (scribbles[0].sum() == 0 and scribbles[1].sum() == 0):
            for i, m in enumerate([gt.astype(np.uint8), (~gt).astype(np.uint8)]):
                if m.sum() < 100:
                    continue
                this_scribble = get_curve_scribble(m, max_srb=5, sort=False)
                scribbles[i] = scribbles[i] | this_scribble

        return scribbles[0].astype(np.uint8), scribbles[1].astype(np.uint8)


if __name__ == '__main__':
    import sys
    mask = cv2.imread(sys.argv[1], cv2.IMREAD_GRAYSCALE)
    gt = cv2.imread(sys.argv[2], cv2.IMREAD_GRAYSCALE)

    fp_scibble, fn_scibble = get_scribble(mask, gt, False)

    cv2.imwrite('s1.png', fp_scibble*255)
    cv2.imwrite('s2.png', fn_scibble*255)