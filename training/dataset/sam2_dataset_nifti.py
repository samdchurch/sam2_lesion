import os
import json
import torch
import numpy as np
import nibabel as nib
import random

from typing import Callable, Iterable, List, Optional, Sequence

from torch.utils.data import BatchSampler, DataLoader, Dataset, IterableDataset, Subset
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets.vision import VisionDataset
from torchvision.transforms import ToPILImage

from training.utils.data_utils import VideoDatapoint, Frame, Object

class NiftiDataset(VisionDataset):
    def __init__(
        self,
        image_folder,
        gt_folder,
        info_folder,
        transforms,
        training,
        max_num_frames,
        file_list = None,
        multiplier = 1,
        multislice = True
        ):
        self.image_folder = image_folder
        self.gt_folder = gt_folder
        self.info_folder = info_folder
        self._transforms = transforms
        self.training = training
        self.max_num_frames = max_num_frames
        self.multislice = multislice
        if file_list is not None:
            with open(file_list) as f:
                self.image_files = f.readlines()
                self.image_files = [file.replace('\n', '') for file in self.image_files]
        else:
            self.image_files = os.listdir(image_folder)

        self.image_files = self.image_files * multiplier


    def _get_datapoint(self, idx):
        file = self.image_files[idx]
        info_file = file.replace('.nii.gz', '.json')
        info_file = os.path.join(self.info_folder, info_file)
        print(info_file, flush=True)
        with open(info_file) as f:
            file_info = json.load(f)

        label_slice_info = file_info['label']
        top_3_slice_info = dict(sorted(label_slice_info.items(), key=lambda item: item[1], reverse=True)[:3])
        #center_slice = int(np.random.choice(list(top_3_slice_info.keys()), p=list(top_3_slice_info.values())))
        center_slice = int(np.random.choice(list(top_3_slice_info.keys())))

        min_slice = int(file_info['image']['min'])
        max_slice = int(file_info['image']['max'])

        image_file = os.path.join(self.image_folder, file)
        label_file = os.path.join(self.gt_folder, file)

        image_data = nib.load(image_file, mmap=True)
        label_data = nib.load(label_file, mmap=True)
        if self.multislice:
            pad = 1
        else:
            pad = 0
        if np.random.rand() < 0.5:
            direction = -1
            end_slice = max(center_slice - self.max_num_frames, min_slice)
            center_slice_image = center_slice + pad
            end_slice_image = end_slice - pad
        else:
            direction = 1
            end_slice = min(center_slice + self.max_num_frames, max_slice)
            center_slice_image = center_slice - pad
            end_slice_image = end_slice + pad
        
        if np.abs(center_slice - end_slice) <= 2:
            return self._get_datapoint(np.random.randint(0, len(self)))
        
        image_slices = image_data.dataobj[:,:,center_slice_image:end_slice_image:direction]
        label_slices = label_data.dataobj[:,:,center_slice:end_slice:direction]

        image_slices = np.array(image_slices)
        label_slices = np.array(label_slices)

        image_slices = image_slices - np.min(image_slices)
        image_slices = image_slices / np.max(image_slices)

        frames = []
        to_pil = ToPILImage()
        for i in range(label_slices.shape[2]):
            # shape (256, 256)
            label_slice = torch.Tensor(label_slices[:,:,i]).squeeze()
            obj = Object(object_id=1, frame_index=i, segment=label_slice)

            make_rgb = False
            if self.multislice:
                image_slice = torch.Tensor(image_slices[:,:,i:i+3]).squeeze()
            else:
                image_slice = torch.Tensor(image_slices[:,:,i])
                image_slice = image_slice.repeat(1, 1, 3)

            image_slice = image_slice.permute(2, 0, 1)
            image_slice = to_pil(image_slice)
            
            # shape (3, 256, 256)
            image_frame = Frame(data=image_slice, objects=[obj])
            frames.append(image_frame)

        h = image_slices.shape[0]
        w = image_slices.shape[1]
        datapoint = VideoDatapoint(frames=frames, video_id=idx, size=(h, w))
        for transform in self._transforms:
            datapoint = transform(datapoint)#, epoch=self.curr_epoch)

        return datapoint

    def __getitem__(self, idx):
        return self._get_datapoint(idx)

    def __len__(self):
        return len(self.image_files)


class TorchTrainNiftiDataset:
    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        num_workers: int,
        shuffle: bool,
        pin_memory: bool,
        drop_last: bool,
        collate_fn: Optional[Callable] = None,
        worker_init_fn: Optional[Callable] = None,

    ) -> None:
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.collate_fn = collate_fn
        self.worker_init_fn = worker_init_fn

        sampler = DistributedSampler(dataset, shuffle=self.shuffle)
        batch_sampler = BatchSampler(sampler, batch_size, drop_last=self.drop_last)

        self.dataloader = DataLoader(
                            dataset,
                            num_workers=self.num_workers,
                            pin_memory=self.pin_memory,
                            batch_sampler=batch_sampler,
                            collate_fn=self.collate_fn,
                            worker_init_fn=self.worker_init_fn)

    def get_loader(self, epoch) -> Iterable:
        return self.dataloader
