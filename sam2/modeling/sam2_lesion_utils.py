import torch
from scipy.spatial.distance import cdist
import numpy as np
import cv2

def get_ortho_point(
        prompt_points, 
        prompt_type, 
        center_slice, 
        ortho_dim,
        point_label=1):

    points = prompt_points[:,:,ortho_dim].squeeze()

    point = int(np.round(np.mean(points.cpu().numpy())))
    center_slice_point = int(np.round((center_slice[0] / 128) * 1024))
    new_point = torch.tensor([[center_slice_point, point]], device=points.device, dtype=points.dtype).unsqueeze(0)
    #new_point = torch.tensor([[point, center_slice_point]], device=points.device, dtype=points.dtype).unsqueeze(0)
    label = torch.tensor([point_label], dtype=torch.int, device=points.device)
    label = label.unsqueeze(0)
    return new_point, label
    

def find_furthest_points_brute(
        masks: torch.Tensor,
        top_percent: float=0.25,
        line_label: int=4):
    
    device = masks.device
    mask = masks[0][0]
    mask = mask.cpu().numpy().astype(np.uint8)
    contours, _ = cv2.findContours(mask, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)



    assert masks.shape[0] == 1
    device = masks.device
    mask = masks[0][0]
    
    indices = torch.nonzero(mask, as_tuple=False)  # Get coordinates of foreground pixels
    if len(indices) < 2:
        points = torch.tensor(np.array([[[0, 0], [0, 0]]]), dtype=torch.float32, device=device)
        labels = torch.tensor([-1, -1], dtype=torch.int, device=device)
        labels = labels.unsqueeze(0)
        return points, labels
    
    contour_points = np.array(contours[0][:, 0, :]).squeeze()

    dist_matrix = cdist(contour_points, contour_points, metric='euclidean')
    #dist_matrix = cdist(indices_np, indices_np, metric='euclidean')  # Compute pairwise distances
    max_idx = np.unravel_index(np.argmax(dist_matrix), dist_matrix.shape)  # Get max dist indices
    #points = torch.stack([indices[max_idx[0]], indices[max_idx[1]]]).unsqueeze(0)

    contour_points = torch.tensor(contour_points, device=device)
    # start_point = torch.flip(contour_points[max_idx[1]], dims=[0])
    # end_point = torch.flip(contour_points[max_idx[0]], dims=[0])
    start_point = contour_points[max_idx[1]]
    end_point = contour_points[max_idx[0]]

    points = torch.stack([start_point, end_point]).unsqueeze(0)

    labels = torch.tensor([line_label, line_label], dtype=torch.int, device=device)
    labels = labels.unsqueeze(0)
    return points, labels

def dice_score(pred: torch.Tensor, target: torch.Tensor, epsilon: float = 1e-6) -> torch.Tensor:
    pred = pred.float()
    target = target.float()
    
    intersection = (pred * target).sum(dim=(1, 2, 3))
    union = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    
    dice = (2. * intersection + epsilon) / (union + epsilon)
    
    return dice.mean()  # Returns the mean Dice score over the batch