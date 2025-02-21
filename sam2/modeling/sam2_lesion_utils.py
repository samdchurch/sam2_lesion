import torch
from scipy.spatial.distance import cdist
import numpy as np

def find_furthest_points_brute(
        masks: torch.Tensor,
        top_percent: float=0.25,
        line_label: int=4):
    assert masks.shape[0] == 1
    device = masks.device
    mask = masks[0][0]

    indices = torch.nonzero(mask, as_tuple=False)  # Get coordinates of foreground pixels
    if len(indices) < 2:
        points = torch.tensor(np.array([[[0, 0], [0, 0]]]), dtype=torch.float32, device=device)
        labels = torch.tensor([line_label, line_label], dtype=torch.int, device=device)
        labels = labels.unsqueeze(0)
        return points, labels
    
    indices_np = indices.cpu().numpy()  # Convert to NumPy for scipy compatibility
    dist_matrix = cdist(indices_np, indices_np, metric='euclidean')  # Compute pairwise distances

    max_idx = np.unravel_index(np.argmax(dist_matrix), dist_matrix.shape)  # Get max dist indices
    points = torch.stack([indices[max_idx[0]], indices[max_idx[1]]]).unsqueeze(0)

    labels = torch.tensor([line_label, line_label], dtype=torch.int, device=device)
    labels = labels.unsqueeze(0)
    return points, labels