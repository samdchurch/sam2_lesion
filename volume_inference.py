import os
import json


import torch


import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
from hydra.core.global_hydra import GlobalHydra
from hydra import initialize_config_module

from skimage import measure


from sam2.modeling.sam2_lesion_utils import find_furthest_points_brute, dice_score
from sam2.build_sam import build_sam2_video_predictor



DATASET_LOCATION = '/app/UserData/public_datasets/ULS23/combined_dataset'
os.environ["HYDRA_FULL_ERROR"] = "1"

def get_mask_range(mask):
    mask = np.array(mask, dtype=bool)
    nonzero = np.any(mask, axis=(0, 1))
    indices = np.where(nonzero)[0]
    return indices[0] ,indices[-1]

def visualize_results(image_file, mask_gt_3d, mask_pred_3d, center_slice, points, anno_type):
    image_data = nib.load(image_file).get_fdata()

    gt_start, gt_end = get_mask_range(mask_gt_3d)
    pred_start, pred_end = get_mask_range(mask_pred_3d)

    start = min(gt_start, pred_start)
    end = max(gt_end, pred_end)

    NUM_TILES = 3

    if anno_type == 'line':
        anno_x_points = [points[0][0][1], points[0][1][1]]
        anno_y_points = [points[0][0][0], points[0][1][0]]

    for slice_num in range(start, end + 1, NUM_TILES):
        num_plots = min(end - slice_num + 1, NUM_TILES)
        
        name = ''
        for i in range(num_plots):
            name = name + str(slice_num + i)
            if i != num_plots - 1:
                name += '_'
            slice_idx = slice_num + i
            image_slice = image_data[:,:,slice_idx]
            image_slice = np.clip(image_slice, a_min=-200, a_max=400)
            mask_slice_gt = np.array(mask_gt_3d[:,:,slice_idx], dtype=np.uint8)
            mask_slice_pred = np.array(mask_pred_3d[:,:,slice_idx], dtype=np.uint8)

            dice_2d = dice_score(torch.tensor(mask_slice_gt).unsqueeze(0).unsqueeze(0), torch.tensor(mask_slice_pred).unsqueeze(0).unsqueeze(0))

            gt_contour = measure.find_contours(mask_slice_gt, level=0.5)
            pred_contour = measure.find_contours(mask_slice_pred, level=0.5)

            plt.subplot(1, num_plots, i + 1)
            plt.imshow(image_slice, cmap='gray')
            if len(gt_contour) > 0:
                gt_contour = gt_contour[0]
                plt.plot(gt_contour[:,1], gt_contour[:,0], c='green', label='GT', alpha=0.5)

            if len(pred_contour) > 0:
                pred_contour = pred_contour[0]
                plt.plot(pred_contour[:,1], pred_contour[:,0], c='orange', label='PRED', alpha=0.5)

            if slice_idx == center_slice:
                plt.plot(anno_y_points, anno_x_points, c='blue', label='ANNO')
            
            plt.title(f'Dice:{dice_2d:02f}')
            plt.axis('off')
            plt.legend()

        
        plt.tight_layout()
        plt.savefig(f'temp_viz/{name}.png', dpi=300)
        plt.close()


    
    # image_slice = image_data[:,:,center_slice]
    # image_slice = np.clip(image_slice, a_min=-200, a_max=400)
    # mask_slice_gt = np.array(mask_gt_3d[:,:,center_slice], dtype=np.uint8)
    # mask_slice_pred = np.array(mask_pred_3d[:,:,center_slice], dtype=np.uint8)



    # plt.imshow(image_slice, cmap='gray')


    # plt.plot(x_points, y_points, c='blue', label='ANNO')
    # plt.legend()
    # plt.savefig('example.png')
    # plt.close()

def predict(predictor, image_file, center_slice, end_slice, points, labels, multislice, mask_pred_3d):
    forward = end_slice > center_slice
    inference_state = predictor.init_state(video_path=image_file, center_frame=center_slice, end_frame=end_slice, multislice=multislice)

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



def predict_volume(predictor, image_file, label_file, info_file, multislice=False, anno_type=None, visualization=False):
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
            multislice=multislice,
            mask_pred_3d=mask_pred_3d)
    
    # backward prediction
    predict(predictor=predictor, 
            image_file=image_file, 
            center_slice=center_slice, 
            end_slice=min_slice, 
            points=points, 
            labels=labels,
            multislice=multislice,
            mask_pred_3d=mask_pred_3d)

    if visualization:
        visualize_results(image_file=image_file, 
                          mask_gt_3d=mask_gt_3d, 
                          mask_pred_3d=mask_pred_3d, 
                          center_slice=center_slice, 
                          points=points, 
                          anno_type=anno_type)
    
    dice_3d = dice_score(pred=mask_pred_3d.unsqueeze(0), target=mask_gt_3d.unsqueeze(0))
    dice_2d = dice_score(pred=mask_pred_3d[:,:,center_slice].unsqueeze(0).unsqueeze(0), target=mask_gt_3d[:,:,center_slice].unsqueeze(0).unsqueeze(0))

    return dice_3d, dice_2d

    

def run_predictor(ckpt_path, model_config, subset_file=None, multislice=False, anno_type=None, visualization=False):
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
    dice_3d_vals = []
    dice_2d_vals = []


    for idx, file in enumerate(files):
        image_file = os.path.join(image_path, file)
        label_file = os.path.join(label_path, file)
        info_file = os.path.join(info_path, file.replace('.nii.gz', '.json'))
        dice_3d, dice_2d = predict_volume(predictor=predictor, 
                       image_file=image_file, 
                       label_file=label_file, 
                       info_file=info_file,
                       multislice=multislice,
                       anno_type=anno_type, 
                       visualization=visualization)
        
        dice_3d_vals.append(dice_3d)
        dice_2d_vals.append(dice_2d)

    print('DICE 3D:', np.mean(dice_3d_vals))
    print(np.std(dice_3d_vals))
    print('DICE 2D:', np.mean(dice_2d_vals))
    print(np.std(dice_2d_vals))

if __name__ == '__main__':
    ckpt_path = '/app/UserData/Sam/sam2_resources/logs/size-tiny_subset-ABDsmall_ep-40_frames-12_baselr-5e-06_visionlr-3e-06_anno-line_affine-50-20_cj-False_gb2_multi-True_lora-False-8_flip/checkpoints/checkpoint.pt'
    size = 't'
    model_config = f'sam2.1_hiera_{size}'
    subset_file = '/app/UserData/Sam/sam2_resources/subsets/ABDsmall_val.txt'
    anno_type = 'line'
    multislice = True
    GlobalHydra.instance().clear()
    initialize_config_module("sam2_resources/config", version_base="1.2")

    run_predictor(ckpt_path=ckpt_path, model_config=model_config, subset_file=subset_file, multislice=multislice, anno_type=anno_type, visualization=False)