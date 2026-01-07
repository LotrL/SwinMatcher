from loguru import logger
import torch
import pytorch_lightning as pl

from src.swinmatcher import SwinMatcher
from src.swinmatcher.utils.supervision import compute_supervision_coarse, compute_supervision_fine
from src.losses.swinmatcher_loss import SwinMatcherLoss
from src.optimizers import build_optimizer, build_scheduler
from src.utils.misc import lower_config, flattenList
from src.utils.profiler import PassThroughProfiler


class PL_SwinMatcher(pl.LightningModule):
    def __init__(self, config, pretrained_ckpt=None, profiler=None, dump_dir=None):
        """
        TODO:
            - use the new version of PL logging API.
        """
        super().__init__()
        # Misc
        self.config = config  # full config
        _config = lower_config(self.config)
        self.swinmatcher_cfg = lower_config(_config['swinmatcher'])
        self.profiler = profiler or PassThroughProfiler()
        self.n_vals_plot = max(config.TRAINER.N_VAL_PAIRS_TO_PLOT // config.TRAINER.WORLD_SIZE, 1)

        # Matcher
        self.matcher = SwinMatcher(config=_config['swinmatcher'])
        self.loss = SwinMatcherLoss(_config)

        # Pretrained weights
        if pretrained_ckpt:
            state_dict = torch.load(pretrained_ckpt, map_location='cpu')['state_dict']
            self.matcher.load_state_dict(state_dict, strict=True)
            logger.info(f"Load \'{pretrained_ckpt}\' as pretrained checkpoint")
        
        # Testing
        self.dump_dir = dump_dir
        
    def configure_optimizers(self):
        # FIXME: The scheduler did not work properly when `--resume_from_checkpoint`
        optimizer = build_optimizer(self, self.config)
        scheduler = build_scheduler(self.config, optimizer)
        return [optimizer], [scheduler]
    
    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx,
                       optimizer_closure, on_tpu, using_native_amp, using_lbfgs):
        # learning rate warm up
        warmup_step = self.config.TRAINER.WARMUP_STEP
        if self.trainer.global_step < warmup_step:
            if self.config.TRAINER.WARMUP_TYPE == 'linear':
                base_lr = self.config.TRAINER.WARMUP_RATIO * self.config.TRAINER.TRUE_LR
                lr = base_lr + \
                    (self.trainer.global_step / self.config.TRAINER.WARMUP_STEP) * \
                    abs(self.config.TRAINER.TRUE_LR - base_lr)
                for pg in optimizer.param_groups:
                    pg['lr'] = lr
            elif self.config.TRAINER.WARMUP_TYPE == 'constant':
                pass
            else:
                raise ValueError(f'Unknown lr warm-up strategy: {self.config.TRAINER.WARMUP_TYPE}')

        # update params
        optimizer.step(closure=optimizer_closure)
        optimizer.zero_grad()

    def _trainval_inference(self, batch):
        with self.profiler.profile("Compute coarse supervision"):
            compute_supervision_coarse(batch, self.config)

        with self.profiler.profile("SwinMatcher"):
            self.matcher(batch)

        with self.profiler.profile("Compute fine supervision"):
            compute_supervision_fine(batch, self.config)

        with self.profiler.profile("Compute losses"):
            self.loss(batch)

    def training_step(self, batch, batch_idx):
        self._trainval_inference(batch)
        
        # logging
        if self.trainer.global_rank == 0 and self.global_step % self.trainer.log_every_n_steps == 0:
            # scalars
            for k, v in batch['loss_scalars'].items():
                self.logger.experiment.add_scalar(f'train/{k}', v, self.global_step)

        return {'loss': batch['loss'],
                'loss_c': batch['loss_c'],
                'loss_f': batch['loss_f'],
                'loss_sub': batch['loss_sub']}

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        avg_loss_c = torch.stack([x['loss_c'] for x in outputs]).mean()
        avg_loss_f = torch.stack([x['loss_f'] for x in outputs]).mean()
        avg_loss_sub = torch.stack([x['loss_sub'] for x in outputs]).mean()

        if self.trainer.global_rank == 0:
            self.logger.experiment.add_scalar('train/epoch_avg_loss', avg_loss, global_step=self.current_epoch)
            self.logger.experiment.add_scalar('train/epoch_avg_loss_c', avg_loss_c, global_step=self.current_epoch)
            self.logger.experiment.add_scalar('train/epoch_avg_loss_f', avg_loss_f, global_step=self.current_epoch)
            self.logger.experiment.add_scalar('train/epoch_avg_loss_sub', avg_loss_sub, global_step=self.current_epoch)

            print(f"Epoch {self.current_epoch}: the average loss is "
                  f"{avg_loss_c.item():.4f} + {avg_loss_f.item():.4f} + {avg_loss_sub.item():.4f} = {avg_loss.item():.4f}")
