import torch
import torch.nn as nn


class SwinMatcherLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config  # config under the global namespace
        self.loss_config = config['swinmatcher']['loss']

        # coarse-level
        self.pos_w = self.loss_config['pos_weight']
        self.neg_w = self.loss_config['neg_weight']
        # fine-level
        self.fine_type = self.loss_config['fine_type']

    def compute_coarse_loss(self, data, weight=None):
        """
        Point-wise CE / Focal Loss with 0 / 1 confidence as gt.
        Args:
        data (dict): {
            conf_matrix_0_to_1 (torch.Tensor): (N, HW0, HW1)
            conf_matrix_1_to_0 (torch.Tensor): (N, HW0, HW1)
            conf_gt (torch.Tensor): (N, HW0, HW1)
            }
            weight (torch.Tensor): (N, HW0, HW1)
        """
        conf_matrix_0_to_1 = data["conf_matrix_0_to_1"]
        conf_matrix_1_to_0 = data["conf_matrix_1_to_0"]
        conf_gt = data["conf_matrix_gt"]

        pos_mask = conf_gt == 1
        c_pos_w = self.pos_w
        # corner case: no gt coarse-level match at all
        if not pos_mask.any():  # assign a wrong gt
            pos_mask[0, 0, 0] = True
            if weight is not None:
                weight[0, 0, 0] = 0.
            c_pos_w = 0.

        conf_matrix_0_to_1 = torch.clamp(conf_matrix_0_to_1, 1e-6, 1 - 1e-6)
        conf_matrix_1_to_0 = torch.clamp(conf_matrix_1_to_0, 1e-6, 1 - 1e-6)
        alpha = self.loss_config['focal_alpha']
        gamma = self.loss_config['focal_gamma']
        # dense supervision
        loss_pos = - alpha * torch.pow(1 - conf_matrix_0_to_1[pos_mask], gamma) * (conf_matrix_0_to_1[pos_mask]).log()
        loss_pos += - alpha * torch.pow(1 - conf_matrix_1_to_0[pos_mask], gamma) * (conf_matrix_1_to_0[pos_mask]).log()
        if weight is not None:
            loss_pos = loss_pos * weight[pos_mask]
        return c_pos_w * loss_pos.mean()

    def compute_fine_loss(self, data):
        """
        Point-wise Focal Loss with 0 / 1 confidence as gt.
        Args:
        data (dict): {
            conf_matrix_fine (torch.Tensor): (N, W_f^2, W_f^2)
            conf_matrix_f_gt (torch.Tensor): (N, W_f^2, W_f^2)
            }
        """
        conf_matrix_fine = data['conf_matrix_fine']
        conf_matrix_f_gt = data['conf_matrix_f_gt']
        pos_mask, neg_mask = conf_matrix_f_gt > 0, conf_matrix_f_gt == 0
        pos_w, neg_w = self.pos_w, self.neg_w

        if not pos_mask.any():  # assign a wrong gt
            pos_mask[0, 0, 0] = True
            pos_w = 0.
        if not neg_mask.any():
            neg_mask[0, 0, 0] = True
            neg_w = 0.

        conf_matrix_fine = torch.clamp(conf_matrix_fine, 1e-6, 1 - 1e-6)
        alpha = self.loss_config['focal_alpha']
        gamma = self.loss_config['focal_gamma']

        loss_pos = - alpha * torch.pow(1 - conf_matrix_fine[pos_mask], gamma) * (conf_matrix_fine[pos_mask]).log()
        # loss_pos *= conf_matrix_f_gt[pos_mask]
        loss_neg = - alpha * torch.pow(conf_matrix_fine[neg_mask], gamma) * (1 - conf_matrix_fine[neg_mask]).log()

        return pos_w * loss_pos.mean() + neg_w * loss_neg.mean()

    def _compute_re_projection_error(self, kpts0, kpts1, T_0to1):
        """
        Args:
        data (dict): {
            kpts0 (torch.Tensor): (N, 2)
            kpts1 (torch.Tensor): (N, 2)
            T_0to1 (torch.Tensor): (N, 3, 3)
            }
        """
        kpts0_homo = torch.cat((kpts0, torch.ones(kpts0.shape[0], 1, device=kpts0.device)), dim=1).unsqueeze(2)  # (N, 3, 1)
        warped_kpts0_homo = torch.matmul(T_0to1, kpts0_homo)  # (N, 3, 1)
        warped_kpts0 = (warped_kpts0_homo[:, :2, :] / warped_kpts0_homo[:, 2:, :]).squeeze(2)  # (N, 2)
        squared_diff = torch.square(warped_kpts0 - kpts1)  # (N, 2)
        distances = torch.sqrt(torch.sum(squared_diff, dim=1))  # (N)
        return distances

    def compute_sub_pixel_loss(self, data):
        """
        Args:
        data (dict): {
            m_bids (torch.Tensor): (N)
            T_0to1 (torch.Tensor): (B, 3, 3)
            mkpts0_f_train (torch.Tensor): (N, 2)
            mkpts1_f_train (torch.Tensor): (N, 2)
            }
        """
        m_bids = data['m_bids']
        kpts0 = data['mkpts0_f_train']
        kpts1 = data['mkpts1_f_train']
        T_0to1 = data['T_0to1'][m_bids]

        re_projection_error = self._compute_re_projection_error(kpts0, kpts1, T_0to1)

        # filter matches with high re-projection error (only train approximately correct fine-level matches)
        loss = re_projection_error[re_projection_error <= 5.0]
        if len(loss) == 0:
            return torch.zeros(1, device=loss.device, requires_grad=False)[0]
        return loss.mean()

    @torch.no_grad()
    def compute_c_weight(self, data):
        """ compute element-wise weights for computing coarse-level loss. """
        if 'mask0' in data:
            c_weight = (data['mask0'].flatten(-2)[..., None] * data['mask1'].flatten(-2)[:, None]).float()
        else:
            c_weight = None
        return c_weight

    def forward(self, data):
        """
        Update:
            data (dict): update{
                'loss': [1] the reduced loss across a batch,
                'loss_scalars' (dict): loss scalars for tensorboard_record
            }
        """
        loss_scalars = {}
        # 0. compute element-wise loss weight
        c_weight = self.compute_c_weight(data)

        # 1. coarse-level loss
        loss_c = self.compute_coarse_loss(data, weight=c_weight)
        loss_c *= self.loss_config['coarse_weight']
        loss = loss_c
        loss_scalars.update({"loss_c": loss_c.clone().detach().cpu()})

        # 2. fine-level matching loss for windows
        loss_f = self.compute_fine_loss(data)
        loss_f *= self.loss_config['fine_weight']
        loss = loss + loss_f
        loss_scalars.update({"loss_f": loss_f.clone().detach().cpu()})

        # 3. sub-pixel refinement loss
        loss_sub = self.compute_sub_pixel_loss(data)
        loss_sub *= self.loss_config['sub_weight']
        loss = loss + loss_sub
        loss_scalars.update({"loss_sub": loss_sub.clone().detach().cpu()})

        loss_scalars.update({'loss': loss.clone().detach().cpu()})
        data.update({"loss": loss,
                     "loss_c": loss_c,
                     "loss_f": loss_f,
                     "loss_sub": loss_sub,
                     "loss_scalars": loss_scalars})
