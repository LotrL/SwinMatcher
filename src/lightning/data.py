from loguru import logger

import pytorch_lightning as pl
from torch import distributed as dist
from torch.utils.data import DataLoader

from src.utils.augment import build_augmentor
from src.datasets.multi_modality import MultiModality


class MultiSceneDataModule(pl.LightningDataModule):
    """
    For distributed training, each training process is assgined
    only a part of the training scenes to reduce memory overhead.
    """

    def __init__(self, args, config):
        super().__init__()

        # 1. data config
        # self.train_data_root = config.DATASET.TRAIN_DATA_ROOT
        self.train_data_root = "/root/autodl-tmp/multi-modal_datasets_homography_30_rotation"

        # 2. dataset config
        # general options
        self.augment_fn = build_augmentor(config.DATASET.AUGMENTATION_TYPE)  # None, options: [None, 'dark', 'mobile']

        # MegaDepth options
        self.mtmd_img_resize = config.DATASET.MGDPT_IMG_RESIZE  # 840
        self.mtmd_img_pad = config.DATASET.MGDPT_IMG_PAD  # True
        self.mtmd_df = config.DATASET.MGDPT_DF  # 8
        self.coarse_scale = 1 / config.SWINMATCHER.RESOLUTION[0]  # 0.125. for training swinmatcher.

        # 3.loader parameters
        self.train_loader_params = {
            'batch_size': args.batch_size,
            'num_workers': args.num_workers,
            'pin_memory': getattr(args, 'pin_memory', True)
        }

        # 4. misc configurations
        self.seed = config.TRAINER.SEED  # 66

    def setup(self, stage=None):
        """
        Setup train dataset. This method will be called by PL automatically.
        Args:
            stage (str): 'fit' in training phase.
        """

        assert stage == 'fit', "stage must be fit"

        try:
            self.world_size = dist.get_world_size()
            self.rank = dist.get_rank()
            logger.info(f"[rank:{self.rank}] world_size: {self.world_size}")
        except AssertionError as ae:
            self.world_size = 1
            self.rank = 0
            logger.warning(str(ae) + " (set world_size = 1 and rank = 0)")

        if stage == 'fit':
            self.train_dataset = MultiModality(self.train_data_root,
                                               mode='train',
                                               img_resize=self.mtmd_img_resize,
                                               df=self.mtmd_df,
                                               img_padding=self.mtmd_img_pad,
                                               augment_fn=self.augment_fn,
                                               coarse_scale=self.coarse_scale)
            logger.info(f'[rank:{self.rank}] Train Dataset loaded!')

    def train_dataloader(self):
        """
        Build training dataloader for MultiModality.
        """
        dataloader = DataLoader(self.train_dataset, **self.train_loader_params)
        return dataloader
