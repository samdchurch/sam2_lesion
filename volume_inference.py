import os
import json
import cv2

import torch


import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
from hydra.core.global_hydra import GlobalHydra
from hydra import initialize_config_module

from sam2.modeling.sam2_lesion_utils import find_furthest_points_brute, dice_score
from sam2.build_sam import build_sam2_video_predictor



DATASET_LOCATION = '/app/UserData/public_datasets/ULS23/combined_dataset'
os.environ["HYDRA_FULL_ERROR"] = "1"

def visualize_results(image_file, mask_gt_3d, mask_pred_3d, center_slice, points, anno_type):
    image_data = nib.load(image_file).get_fdata()
    
    image_slice = image_data[:,:,center_slice]
    image_slice = np.clip(image_slice, a_min=-200, a_max=400)
    mask_slice_gt = np.array(mask_gt_3d[:,:,center_slice], dtype=np.uint8)
    mask_slice_pred = np.array(mask_pred_3d[:,:,center_slice], dtype=np.uint8)
    #print(np.nonzero(mask_slice_pred))
    gt_contour, _ = cv2.findContours(mask_slice_gt, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    pred_contour, _ = cv2.findContours(mask_slice_pred, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    gt_contour = gt_contour[0]
    pred_contour = pred_contour[0]
    print(gt_contour.shape)
    print(pred_contour.shape)
    plt.imshow(image_slice, cmap='gray')
    if anno_type == 'line':
        x_points = [points[0][0][1], points[0][1][1]]
        y_points = [points[0][0][0], points[0][1][0]]

    plt.imshow(mask_slice_gt, alpha=0.2)
    plt.imshow(mask_slice_pred, alpha=0.2)
    plt.plot(x_points, y_points)
    plt.plot(gt_contour[:,0,0], gt_contour[:,0,1])
    plt.plot(pred_contour[:,0,0], pred_contour[:,0,1])
    plt.savefig('example.png')

def predict(predictor, image_file, center_slice, end_slice, points, labels, mask_pred_3d):
    forward = end_slice > center_slice
    inference_state = predictor.init_state(video_path=image_file, center_frame=center_slice, end_frame=end_slice)

    _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
        inference_state=inference_state,
        frame_idx=0,
        obj_id=1,
        points=points,
        labels=labels,
    )
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
        pred_mask = (out_mask_logits[0] > 0.0).squeeze()
        if torch.all(pred_mask == 0):
            break
        if forward:
            mask_pred_3d[:,:,center_slice + out_frame_idx] = pred_mask
        else:
            mask_pred_3d[:,:,center_slice - out_frame_idx] = pred_mask



def predict_volume(predictor, image_file, label_file, info_file, anno_type=None, visualization=False):
    with open(info_file) as f:
        image_info = json.load(f)
    
    min_slice = image_info['image']['min']
    max_slice = image_info['image']['max']
    center_slice = int(max(image_info['label'], key=image_info['label'].get))
    mask_gt_3d = torch.tensor(nib.load(label_file).get_fdata()).squeeze()
    mask_pred_3d = torch.zeros_like(mask_gt_3d)
    center_slice_mask = mask_gt_3d[:,:,center_slice].unsqueeze(0).unsqueeze(0)
    if anno_type == 'line':
        points, labels = find_furthest_points_brute(center_slice_mask)
    
    # forward prediction
    predict(predictor=predictor, 
            image_file=image_file, 
            center_slice=center_slice, 
            end_slice=max_slice, 
            points=points, 
            labels=labels, 
            mask_pred_3d=mask_pred_3d)
    
    # backward prediction
    predict(predictor=predictor, 
            image_file=image_file, 
            center_slice=center_slice, 
            end_slice=min_slice, 
            points=points, 
            labels=labels, 
            mask_pred_3d=mask_pred_3d)

    if visualization:
        visualize_results(image_file=image_file, 
                          mask_gt_3d=mask_gt_3d, 
                          mask_pred_3d=mask_pred_3d, 
                          center_slice=center_slice, 
                          points=points, 
                          anno_type=anno_type)
    
    dice_3d = dice_score(pred=mask_pred_3d.unsqueeze(0), target=mask_gt_3d.unsqueeze(0))
    print('DICE:', dice_3d)

    

def run_predictor(ckpt_path, model_config, subset_file=None, anno_type=None, visualization=False):
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    
    predictor = build_sam2_video_predictor(model_config, ckpt_path, device=device)

    if subset_file is None or 'val' in subset_file:
        dataset_path = os.path.join(DATASET_LOCATION, 'val')
    else:
        dataset_path = os.path.join(DATASET_LOCATION, 'train')
    
    image_path = os.path.join(dataset_path, 'images')
    info_path = os.path.join(dataset_path, 'info')
    label_path = os.path.join(dataset_path, 'labels')

    if subset_file is not None:
        with open(subset_file) as f:
            files = f.readlines()
        files = [file.replace('\n', '') for file in files]
    else:
        files = os.listdir(image_path)

    for file in files:
        image_file = os.path.join(image_path, file)
        label_file = os.path.join(label_path, file)
        info_file = os.path.join(info_path, file.replace('.nii.gz', '.json'))
        predict_volume(predictor=predictor, 
                       image_file=image_file, 
                       label_file=label_file, 
                       info_file=info_file, 
                       anno_type=anno_type, 
                       visualization=visualization)
        break

if __name__ == '__main__':
    ckpt_path = '/app/UserData/Sam/sam2_resources/logs/size-tiny_subset-ABDsmall_ep-40_frames-12_baselr-5e-06_visionlr-3e-06_anno-line_affine-50-20_cj-False_gb2_multi-True_lora-False-8/checkpoints/checkpoint.pt'
    size = 't'
    model_config = f'sam2.1_hiera_{size}'
    subset_file = '/app/UserData/Sam/sam2_resources/subsets/ABDsmall_val.txt'
    anno_type = 'line'
    GlobalHydra.instance().clear()
    initialize_config_module("sam2_resources/config", version_base="1.2")

    run_predictor(ckpt_path=ckpt_path, model_config=model_config, subset_file=subset_file, anno_type=anno_type, visualization=True)