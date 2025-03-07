from typing import Any, Optional, Tuple, List
import torch
from torch import nn
import math

class PositionEmbeddingSine3D(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention Is All You Need paper, generalized to work on images.
    """

    def __init__(
        self,
        num_pos_feats,
        temperature: int = 10000,
        normalize: bool = True,
        scale: Optional[float] = None,
        # Following settings only relevant
        # for warmping up cache for compilation
        warmup_cache: bool = True,
        image_size: int = 1024,
        strides: Tuple[int] = (4, 8, 16, 32),
    ):
        super().__init__()
        assert num_pos_feats % 2 == 0, "Expecting even model width"
        self.num_pos_feats = num_pos_feats // 2
        self.num_x_feats, self.num_y_feats, self.num_z_feats = self.divide_into_three(num_pos_feats)
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

        self.cache = {}
        if warmup_cache and torch.cuda.is_available():
            # Warmup cache for cuda, to help with compilation
            device = torch.device("cuda")
            for stride in strides:
                cache_key = (image_size // stride, image_size // stride)
                self._pe(1, device, *cache_key)

    def divide_into_three(self, n):
        a = n // 3
        b = a
        c = n - (2 * a)  # Ensure the sum is still n

        # Make sure c is the smallest
        if c > a:
            a += 1
            b += 1
            c -= 2
        
        assert a + b + c == n

        return a, b, c
    
    def _encode_xy(self, x, y):
        # The positions are expected to be normalized
        assert len(x) == len(y) and x.ndim == y.ndim == 1
        x_embed = x * self.scale
        y_embed = y * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, None] / dim_t
        pos_y = y_embed[:, None] / dim_t
        pos_x = torch.stack(
            (pos_x[:, 0::2].sin(), pos_x[:, 1::2].cos()), dim=2
        ).flatten(1)
        pos_y = torch.stack(
            (pos_y[:, 0::2].sin(), pos_y[:, 1::2].cos()), dim=2
        ).flatten(1)
        return pos_x, pos_y

    @torch.no_grad()
    def encode_boxes(self, x, y, w, h):
        pos_x, pos_y = self._encode_xy(x, y)
        pos = torch.cat((pos_y, pos_x, h[:, None], w[:, None]), dim=1)
        return pos

    encode = encode_boxes  # Backwards compatibility

    @torch.no_grad()
    def encode_points(self, x, y, labels):
        (bx, nx), (by, ny), (bl, nl) = x.shape, y.shape, labels.shape
        assert bx == by and nx == ny and bx == bl and nx == nl
        pos_x, pos_y = self._encode_xy(x.flatten(), y.flatten())
        pos_x, pos_y = pos_x.reshape(bx, nx, -1), pos_y.reshape(by, ny, -1)
        pos = torch.cat((pos_y, pos_x, labels[:, :, None]), dim=2)
        return pos
    
    @torch.no_grad()
    def _pe_3d(self, input_shape, fixed_dims, fixed_slices, device):
        H, W = input_shape[-2], input_shape[-1]
        B = input_shape[0]
        all_pos = []
        for i in range(B):
            fixed_dim = fixed_dims[i]
            fixed_slice = fixed_slices[i]

            if fixed_dim == 0:
                z_embed = (
                    torch.arange(1, H + 1, dtype=torch.float32, device=device)
                    .view(1, -1, 1)
                    .repeat(1, 1, W)
                )
                x_embed = (
                    torch.arange(1, W + 1, dtype=torch.float32, device=device)
                    .view(1, 1, -1)
                    .repeat(1, H, 1)
                )
                y_embed = torch.full_like(x_embed, fixed_slice)
            elif fixed_dim == 1:
                y_embed = (
                    torch.arange(1, H + 1, dtype=torch.float32, device=device)
                    .view(1, -1, 1)
                    .repeat(1, 1, W)
                )
                z_embed = (
                    torch.arange(1, W + 1, dtype=torch.float32, device=device)
                    .view(1, 1, -1)
                    .repeat(1, H, 1)
                )
                x_embed = torch.full_like(z_embed, fixed_slice)
            if fixed_dim == 2:
                y_embed = (
                    torch.arange(1, H + 1, dtype=torch.float32, device=device)
                    .view(1, -1, 1)
                    .repeat(1, 1, W)
                )
                x_embed = (
                    torch.arange(1, W + 1, dtype=torch.float32, device=device)
                    .view(1, 1, -1)
                    .repeat(1, H, 1)
                )
                z_embed = torch.full_like(x_embed, fixed_slice)


            dim_t = torch.arange(self.num_y_feats, dtype=torch.float32, device=device)
            dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_y_feats)
            pos_y = y_embed[:, :, :, None] / dim_t


            dim_t = torch.arange(self.num_x_feats, dtype=torch.float32, device=device)
            dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_x_feats)
            pos_x = x_embed[:, :, :, None] / dim_t

            dim_t = torch.arange(self.num_z_feats, dtype=torch.float32, device=device)
            dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_z_feats)
            pos_z = z_embed[:, :, :, None] / dim_t

            pos_x = torch.stack(
                (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4
            ).flatten(3)
            pos_y = torch.stack(
                (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4
            ).flatten(3)
            pos_z = torch.stack(
                (pos_z[:, :, :, 0::2].sin(), pos_z[:, :, :, 1::2].cos()), dim=4
            ).flatten(3)
            pos = torch.cat((pos_y, pos_x, pos_z), dim=3).permute(0, 3, 1, 2)
            all_pos.append(pos)
        all_pos = torch.stack(all_pos)
        all_pos = all_pos.squeeze(1)

        return all_pos

    @torch.no_grad()
    def _pe(self, B, device, *cache_key):
        H, W = cache_key

        if cache_key in self.cache:
            return self.cache[cache_key].to(device)[None].repeat(B, 1, 1, 1)

        y_embed = (
            torch.arange(1, H + 1, dtype=torch.float32, device=device)
            .view(1, -1, 1)
            .repeat(B, 1, W)
        )
    
        x_embed = (
            torch.arange(1, W + 1, dtype=torch.float32, device=device)
            .view(1, 1, -1)
            .repeat(B, H, 1)
        )

        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)
        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        self.cache[cache_key] = pos[0]

        return pos

    @torch.no_grad()
    def forward(self, 
                x: torch.Tensor,
                fixed_dims: List[int],
                fixed_slices: List[int]):
        

        B = x.shape[0]
        cache_key = (x.shape[-2], x.shape[-1])

        return self._pe_3d(x.shape, fixed_dims, fixed_slices, x.device)