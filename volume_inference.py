import os
import json

import torch

import nibabel as nib
from hydra.core.global_hydra import GlobalHydra
from hydra import initialize_config_module

from sam2.modeling.sam2_lesion_utils import find_furthest_points_brute
from sam2.build_sam import build_sam2_video_predictor


DATASET_LOCATION = '/app/UserData/public_datasets/ULS23/combined_dataset'
os.environ["HYDRA_FULL_ERROR"] = "1"


def predict_volume(predictor, image_file, label_file, info_file, anno_type=None, visualization=False):
    with open(info_file) as f:
        image_info = json.load(f)
    
    min_slice = image_info['image']['min']
    max_slice = image_info['image']['max']
    center_slice = int(max(image_info['label'], key=image_info['label'].get))
    mask_3d = torch.tensor(nib.load(label_file).get_fdata())
    center_slice_mask = mask_3d[:,:,center_slice].squeeze().unsqueeze(0)
    if anno_type == 'line':
        points, labels = find_furthest_points_brute(center_slice_mask)

    inference_state = predictor.init_state(video_path=image_file, center_frame=center_slice, end_frame=max_slice)

    _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
        inference_state=inference_state,
        frame_idx=0,
        obj_id=1,
        points=points,
        labels=labels,
    )

    video_segments = {}
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
        pred_mask = out_mask_logits[0] > 0.0
        if torch.all(pred_mask == 0):
            break
        print(out_frame_idx)

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
        predict_volume(predictor=predictor, image_file=image_file, label_file=label_file, info_file=info_file, anno_type=anno_type, visualization=visualization)

if __name__ == '__main__':
    ckpt_path = '/app/UserData/Sam/sam2_resources/logs/size-tiny_subset-ABDsmall_ep-40_frames-12_baselr-5e-06_visionlr-3e-06_anno-line_affine-50-20_cj-False_gb2_multi-True_lora-False-8/checkpoints/checkpoint.pt'
    size = 't'
    model_config = f'sam2.1_hiera_{size}'
    subset_file = '/app/UserData/Sam/sam2_resources/subsets/ABDsmall_val.txt'
    anno_type = 'line'
    GlobalHydra.instance().clear()
    initialize_config_module("sam2_resources/config", version_base="1.2")

    run_predictor(ckpt_path=ckpt_path, model_config=model_config, subset_file=subset_file, anno_type=anno_type)