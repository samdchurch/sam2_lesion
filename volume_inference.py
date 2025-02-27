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

def all_views_viz(image_data, mask_gt_3d, mask_pred_3d, save_dir, dice_3d, image_name):
    save_dir = os.path.join(save_dir, 'all_views')
    os.makedirs(save_dir, exist_ok=True)

    mask_gt_3d = np.array(mask_gt_3d)
    sag_dim = np.argmax(np.sum(mask_gt_3d, axis=(1, 2)))
    cor_dim = np.argmax(np.sum(mask_gt_3d, axis=(0, 2)))
    ax_dim = np.argmax(np.sum(mask_gt_3d, axis=(0, 1)))
    image_data = np.clip(image_data, -100, 400)
    plt.subplot(1, 3, 1)
    ax_slice = np.rot90(image_data[:,:,ax_dim])
    mask_slice_gt = np.rot90(mask_gt_3d[:, :, ax_dim])
    mask_slice_pred = np.rot90(mask_pred_3d[:, :, ax_dim])
    gt_contour = measure.find_contours(mask_slice_gt, level=0.5)
    pred_contour = measure.find_contours(mask_slice_pred, level=0.5)
    if len(gt_contour) > 0:
        gt_contour = gt_contour[0]
        plt.plot(gt_contour[:,1], gt_contour[:,0], c='green', label='GT', alpha=0.5)

    if len(pred_contour) > 0:
        pred_contour = pred_contour[0]
        plt.plot(pred_contour[:,1], pred_contour[:,0], c='orange', label='PRED', alpha=0.5)
    plt.imshow(ax_slice, cmap='gray')
    plt.axis('off')
    

    plt.subplot(1, 3, 2)
    cor_slice = np.rot90(image_data[:,cor_dim,:])
    mask_slice_gt = np.rot90(mask_gt_3d[:, cor_dim, :])
    mask_slice_pred = np.rot90(mask_pred_3d[:, cor_dim, :])
    gt_contour = measure.find_contours(mask_slice_gt, level=0.5)
    pred_contour = measure.find_contours(mask_slice_pred, level=0.5)
    plt.imshow(cor_slice, cmap='gray')
    plt.axis('off')
    plt.gca().set_aspect(2)


    if len(gt_contour) > 0:
        gt_contour = gt_contour[0]
        plt.plot(gt_contour[:,1], gt_contour[:,0], c='green', label='GT', alpha=0.5)

    if len(pred_contour) > 0:
        pred_contour = pred_contour[0]
        plt.plot(pred_contour[:,1], pred_contour[:,0], c='orange', label='PRED', alpha=0.5)

    plt.subplot(1, 3, 3)
    sag_slice = np.rot90(image_data[sag_dim, :, :])
    mask_slice_gt = np.rot90(mask_gt_3d[sag_dim, :, :])
    mask_slice_pred = np.rot90(mask_pred_3d[sag_dim, :, :])
    gt_contour = measure.find_contours(mask_slice_gt, level=0.5)
    pred_contour = measure.find_contours(mask_slice_pred, level=0.5)
    plt.imshow(sag_slice, cmap='gray')
    plt.gca().set_aspect(2)

    plt.axis('off')

    if len(gt_contour) > 0:
        gt_contour = gt_contour[0]
        plt.plot(gt_contour[:,1], gt_contour[:,0], c='green', label='GT', alpha=0.5)

    if len(pred_contour) > 0:
        pred_contour = pred_contour[0]
        plt.plot(pred_contour[:,1], pred_contour[:,0], c='orange', label='PRED', alpha=0.5)
    plt.axis('off')
    plt.legend(fontsize=10)

    plt.tight_layout()
    
    plt.suptitle(f'DICE:{dice_3d:.2f}')
    plt.savefig(f'{save_dir}/{image_name}.png', dpi=500)
    plt.close()

def visualize_results(image_file, mask_gt_3d, mask_pred_3d, center_slice, points, anno_type, dice_3d):
    image_name = image_file.split('/')[-1].split('.')[0]
    image_data = nib.load(image_file).get_fdata()

    gt_start, gt_end = get_mask_range(mask_gt_3d)
    pred_start, pred_end = get_mask_range(mask_pred_3d)

    start = min(gt_start, pred_start)
    end = max(gt_end, pred_end)

    NUM_TILES = 3

    if dice_3d <= 0.5:
        quality_set = 'bad'
    elif dice_3d <= 0.65:
        quality_set = 'poor'
    elif dice_3d < 0.8:
        quality_set = 'good'
    else:
        quality_set = 'great'

    save_dir = f'temp_viz/{quality_set}'
    os.makedirs(save_dir, exist_ok=True)

    all_views_viz(image_data=image_data, mask_gt_3d=mask_gt_3d, mask_pred_3d=mask_pred_3d, save_dir=save_dir, dice_3d=dice_3d, image_name=image_name)

    save_dir = f'temp_viz/{quality_set}/all_frames/{image_name}'
    os.makedirs(save_dir, exist_ok=True)
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

        plt.suptitle(f'Dice 3D:{dice_3d:02f}')
        
        plt.tight_layout()

        plt.savefig(f'{save_dir}/{name}.png', dpi=300)
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


        
        
    pred_forward = mask_pred_3d[:,:,center_slice:]
    gt_forward = mask_gt_3d[:,:,center_slice:]
    loc_pred = np.where(pred_forward)[2]
    loc_gt = np.where(gt_forward)[2]

    forward_pred_num_slices = np.max(loc_pred) - np.min(loc_pred) + 1
    forward_gt_num_slices = np.max(loc_gt) - np.min(loc_gt) + 1


    pred_backward = mask_pred_3d[:,:,:center_slice+1]
    gt_backward = mask_gt_3d[:,:,:center_slice+1]

    loc_pred = np.where(pred_backward)[2]
    loc_gt = np.where(gt_backward)[2]

    backward_pred_num_slices = np.max(loc_pred) - np.min(loc_pred) + 1
    backward_gt_num_slices = np.max(loc_gt) - np.min(loc_gt) + 1

    num_gt_slices = [forward_gt_num_slices, backward_gt_num_slices]
    num_diff = [forward_pred_num_slices - forward_gt_num_slices, backward_pred_num_slices - backward_gt_num_slices]

    dice_forward = dice_score(pred=pred_forward.unsqueeze(0), target=gt_forward.unsqueeze(0))
    dice_backward = dice_score(pred=pred_backward.unsqueeze(0), target=gt_backward.unsqueeze(0))
    
    dice_3d = dice_score(pred=mask_pred_3d.unsqueeze(0), target=mask_gt_3d.unsqueeze(0))
    dice_2d = dice_score(pred=mask_pred_3d[:,:,center_slice].unsqueeze(0).unsqueeze(0), target=mask_gt_3d[:,:,center_slice].unsqueeze(0).unsqueeze(0))

    if visualization:
        visualize_results(image_file=image_file, 
                          mask_gt_3d=mask_gt_3d, 
                          mask_pred_3d=mask_pred_3d, 
                          center_slice=center_slice, 
                          points=points, 
                          anno_type=anno_type,
                          dice_3d=dice_3d)

    return dice_3d, dice_2d, num_gt_slices, num_diff

    

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
    all_gt_num = []
    all_diff = []
    for idx, file in enumerate(files):
        image_file = os.path.join(image_path, file)
        label_file = os.path.join(label_path, file)
        info_file = os.path.join(info_path, file.replace('.nii.gz', '.json'))
        dice_3d, dice_2d, num_gt_slices, num_diff = predict_volume(predictor=predictor, 
                       image_file=image_file, 
                       label_file=label_file, 
                       info_file=info_file,
                       multislice=multislice,
                       anno_type=anno_type, 
                       visualization=visualization)
        
        dice_3d_vals.append(dice_3d)
        dice_2d_vals.append(dice_2d)

        for num in num_gt_slices:
            all_gt_num.append(num)
        for num in num_diff:
            all_diff.append(num)

    print('DICE 3D:', np.mean(dice_3d_vals))
    print(np.std(dice_3d_vals))
    print('DICE 2D:', np.mean(dice_2d_vals))
    print(np.std(dice_2d_vals))

    plt.scatter(all_gt_num, all_diff)
    plt.savefig('example.png')
    plt.close()

    plt.boxplot(dice_3d_vals)
    plt.savefig('dice_3d.png')
    plt.close()
    plt.boxplot(dice_2d_vals)
    plt.savefig('dice_2d.png')
    plt.close()

if __name__ == '__main__':
    ckpt_path = '/app/UserData/Sam/sam2_resources/logs/size-tiny_subset-ABD_ep-40_frames-12_baselr-5e-06_visionlr-3e-06_anno-line_affine-50-20_cj-False_gb2_multi-False_lora-False-8/checkpoints/checkpoint.pt'
    size = 't'
    model_config = f'sam2.1_hiera_{size}'
    subset_file = '/app/UserData/Sam/sam2_resources/subsets/ABD_val.txt'
    anno_type = 'line'
    multislice = False
    GlobalHydra.instance().clear()
    initialize_config_module("sam2_resources/config", version_base="1.2")

    run_predictor(ckpt_path=ckpt_path, model_config=model_config, subset_file=subset_file, multislice=multislice, anno_type=anno_type, visualization=True)