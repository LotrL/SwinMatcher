import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import os
import random
import numpy as np

from src.utils.dataset import read_multi_modality_gray


class MultiModality(Dataset):
    def __init__(self,
                 root_dir,
                 mode='train',
                 img_resize=None,
                 df=None,
                 img_padding=False,
                 augment_fn=None,
                 **kwargs):
        """
        Manage one scene(npz_path) of Multi-Modality dataset.

        Args:
            root_dir (str): multi-modality root directory that has `phoenix`.
            mode (str): options are ['train', 'val', 'test'].
            img_resize (int, optional): the longer edge of resized images. None for no resize. 640 is recommended.
                                        This is useful during training with batches and testing with memory intensive algorithms.
            df (int, optional): image size division factor. NOTE: this will change the final image size after img_resize.
            img_padding (bool): If set to 'True', zero-pad the image to squared size. This is useful during training.
            augment_fn (callable, optional): augments images with pre-defined visual effects.
        """
        super().__init__()
        self.root_dir = root_dir
        self.mode = mode
        self.dataset = []
        self.build_dataset()

        # parameters for image resizing and padding
        # if mode == 'train':
        #     assert img_resize is not None and img_padding
        self.img_resize = img_resize
        self.df = df
        self.img_padding = img_padding

        # for training LoFTR
        self.augment_fn = augment_fn if mode == 'train' else None
        self.coarse_scale = getattr(kwargs, 'coarse_scale', 0.125)

    def build_dataset(self):
        scenes = sorted(os.listdir(self.root_dir))
        for scene in scenes:
            scene_path = self.root_dir + f"/{scene}"
            scene_pairs = sorted(os.listdir(scene_path))
            for scene_pair in scene_pairs:
                scene_pair_path = scene_path + f"/{scene_pair}"
                folders = sorted(os.listdir(scene_pair_path))
                for folder in folders:
                    folder_path = scene_pair_path + f"/{folder}"
                    files = sorted(os.listdir(folder_path))
                    for i in range(3, len(files)):  # pass the first pair
                        if i % 2 == 1:
                            self.dataset.append({"image0_path": f"{folder_path}/{files[0]}",  # files[1] -> files[0]
                                                 "image1_path": f"{folder_path}/{files[i]}",
                                                 "affine_matrix_path": f"{folder_path}/{files[i + 1]}"})
        np.random.seed(42)
        np.random.shuffle(self.dataset)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]

        # TODO: Support augmentation & handle seeds for each worker correctly.
        # ----- pretrain -----
        thermal_option0, thermal_option1 = False, False
        # ----- finetune -----
        # if random.choice([0, 1]) == 0:
        #     thermal_option0, thermal_option1 = True, False
        # else:
        #     thermal_option0, thermal_option1 = False, True

        # noinspection PyTypeChecker
        image0, mask0, scale0 = read_multi_modality_gray(
            item["image0_path"], self.img_resize, self.df, self.img_padding, None, thermal_option0)
        # np.random.choice([self.augment_fn, None], p=[0.5, 0.5]))

        # noinspection PyTypeChecker
        image1, mask1, scale1 = read_multi_modality_gray(
            item["image1_path"], self.img_resize, self.df, self.img_padding, None, thermal_option1)
        # np.random.choice([self.augment_fn, None], p=[0.5, 0.5]))

        # read and compute relative poses
        # noinspection PyTypeChecker
        T_0to1 = torch.from_numpy(np.loadtxt(item["affine_matrix_path"]).astype(np.float32))  # (3, 3)
        T_1to0 = T_0to1.inverse()

        data = {
            'image0': image0,  # (1, h, w)
            'image1': image1,
            'T_0to1': T_0to1,  # (3, 3)
            'T_1to0': T_1to0,
            'scale0': scale0,  # [scale_w, scale_h]
            'scale1': scale1,
            'dataset_name': 'Multi-Modality',
            'pair_id': idx,
        }

        # for LoFTR training
        if mask0 is not None:  # img_padding is True
            if self.coarse_scale:
                [ts_mask_0, ts_mask_1] = F.interpolate(torch.stack([mask0, mask1], dim=0)[None].float(),
                                                       scale_factor=self.coarse_scale,
                                                       mode='nearest',
                                                       recompute_scale_factor=False)[0].bool()
            data.update({'mask0': ts_mask_0, 'mask1': ts_mask_1})

        return data
