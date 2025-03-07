import os
import json
import torch
import numpy as np
import nibabel as nib
import random
import cv2

from typing import Callable, Iterable, List, Optional, Sequence

from torch.utils.data import BatchSampler, DataLoader, Dataset, IterableDataset, Subset
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets.vision import VisionDataset
from torchvision.transforms import ToPILImage

from training.utils.data_utils import VideoDatapoint, Frame, Object, NiftiDatapoint

from collections import Counter

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
        multislice = True,
        num_ortho_slices = 0
        ):
        self.image_folder = image_folder
        self.gt_folder = gt_folder
        self.info_folder = info_folder
        self._transforms = transforms
        self.training = training
        self.max_num_frames = max_num_frames
        self.multislice = multislice
        self.num_ortho_slices = num_ortho_slices
        if file_list is not None:
            with open(file_list) as f:
                self.image_files = f.readlines()
                self.image_files = [file.replace('\n', '') for file in self.image_files]
        else:
            self.image_files = os.listdir(image_folder)
        np.random.shuffle(self.image_files)
        self.image_files = self.image_files * multiplier


    def _get_datapoint(self, idx):
        file = self.image_files[idx]
        info_file = file.replace('.nii.gz', '.json')
        info_file = os.path.join(self.info_folder, info_file)
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

        image_data = nib.load(image_file)
        label_data = nib.load(label_file)
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
        
        if self.num_ortho_slices == 0:
            image_slices = image_data.dataobj[:,:,center_slice_image:end_slice_image:direction]
            label_slices = label_data.dataobj[:,:,center_slice:end_slice:direction]
            image_slices = torch.tensor(image_slices.copy(), device=torch.device("cuda"))
            label_slices = torch.tensor(label_slices.copy(), device=torch.device("cuda"))
            fixed_slices = []
            fixed_dims = []
            for j in range(center_slice, end_slice, direction):
                fixed_dims.append(2)
                fixed_slices.append(j)
        if self.num_ortho_slices == 2:
            axial_slice = image_data.dataobj[:,:,center_slice_image].squeeze()
            axial_label = label_data.dataobj[:,:,center_slice].squeeze()
            top_num = 3
            loc = np.where(axial_label)

            counter = Counter(loc[0])
            first_dim_loc = counter.most_common(top_num)
            first_dim = np.random.choice([idx for idx, count in first_dim_loc])

            counter = Counter(loc[1])
            second_dim_loc = counter.most_common(top_num)
            second_dim = np.random.choice([idx for idx, count in second_dim_loc])

            sag_slice = image_data.dataobj[first_dim,:,:].squeeze()
            sag_label = label_data.dataobj[first_dim,:,:].squeeze()
            sag_slice = cv2.resize(sag_slice, axial_slice.shape, interpolation=cv2.INTER_LINEAR)
            sag_label = cv2.resize(sag_label, axial_label.shape, interpolation=cv2.INTER_LINEAR)

            cor_slice = image_data.dataobj[:,second_dim,:].squeeze()
            cor_label = label_data.dataobj[:,second_dim,:].squeeze()
            cor_slice = cv2.resize(cor_slice, axial_slice.shape, interpolation=cv2.INTER_LINEAR)
            cor_label = cv2.resize(cor_label, axial_label.shape, interpolation=cv2.INTER_LINEAR)

            slice_list = []
            label_list = []
            fixed_slices = []
            fixed_dims = []

            fixed_dims.append(2)
            fixed_dims.append(0)
            fixed_dims.append(1)

            fixed_slices.append(center_slice)
            fixed_slices.append(first_dim)
            fixed_slices.append(second_dim)

            slice_list.append(torch.tensor(axial_slice))
            slice_list.append(torch.tensor(sag_slice))
            slice_list.append(torch.tensor(cor_slice))
            label_list.append(torch.tensor(axial_label))
            label_list.append(torch.tensor(sag_label))
            label_list.append(torch.tensor(cor_label))

            if direction == -1:
                next_slice = center_slice_image - 1
            else:
                next_slice = center_slice_image + 1
            for j in range(next_slice, end_slice_image, direction):
                slice_list.append(torch.tensor(image_data.dataobj[:,:,j]).squeeze())
            for j in range(next_slice, end_slice, direction):
                label_list.append(torch.tensor(label_data.dataobj[:,:,j]).squeeze())
                fixed_slices.append(j)
                fixed_dims.append(2)


            slice_list = torch.stack(slice_list)
            label_list = torch.stack(label_list)

            image_slices = slice_list.permute(1, 2, 0).unsqueeze(-1)
            label_slices = label_list.permute(1, 2, 0).unsqueeze(-1)
            image_slices.to(torch.device('cuda'))
            label_slices.to(torch.device('cuda'))

        image_slices = image_slices - torch.min(image_slices)
        image_slices = image_slices / torch.max(image_slices)
        frames = []
        #to_pil = ToPILImage()
        
        for i in range(label_slices.shape[2]):
            # shape (256, 256)
            label_slice = label_slices[:,:,i].squeeze()
            obj = Object(object_id=1, frame_index=i, segment=label_slice)

            if self.multislice:
                image_slice = image_slices[:,:,i:i+3].squeeze()
            else:
                image_slice = image_slices[:,:,i]
                image_slice = image_slice.repeat(1, 1, 3)

            image_slice = image_slice.permute(2, 0, 1)
            #image_slice = to_pil(image_slice)
            
            # shape (3, 256, 256)
            image_frame = Frame(data=image_slice, objects=[obj])
            frames.append(image_frame)

        h = image_slices.shape[0]
        w = image_slices.shape[1]
        datapoint = NiftiDatapoint(frames=frames, video_id=idx, size=(h, w), fixed_slices=fixed_slices, fixed_dims=fixed_dims)
        for transform in self._transforms:
            datapoint = transform(datapoint)

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
                            persistent_workers=True,
                            worker_init_fn=self.worker_init_fn)

    def get_loader(self, epoch) -> Iterable:
        return self.dataloader
