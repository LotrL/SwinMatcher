import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.einops import rearrange


class FineMatching(nn.Module):
    """FineMatching with s2d paradigm"""

    def __init__(self, config):
        super().__init__()
        self.temperature = config['fine']['dsmax_temperature']
        self.W_f = config['fine_window_size']
        self.fine_thr = config['fine']['thr']
        # dim_f = config['resnetfpn']['block_dims'][0]
        dim_f = 96
        self.subpixel_conv_block = nn.Sequential(
            nn.Conv2d(2 * dim_f, 2 * dim_f, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(2 * dim_f, 2 * dim_f, kernel_size=3, stride=3, bias=False),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(2 * dim_f, 4, kernel_size=1, stride=1, bias=False)
        )

    def forward(self, feat_f0_unfold, feat_f1_unfold, feat_f0_whole, feat_f1_whole, data):
        """
        Args:
            feat_f0_unfold (torch.Tensor): [M, WW, C]
            feat_f1_unfold (torch.Tensor): [M, WW, C]
            feat_f0_whole (torch.Tensor): [N, C, H0_f, W0_f]
            feat_f1_whole (torch.Tensor): [N, C, H1_f, W1_f]
            data (dict)
        Update:
            data (dict):{
                'expec_f' (torch.Tensor): [M, 3],
                'mkpts0_f' (torch.Tensor): [M, 2],
                'mkpts1_f' (torch.Tensor): [M, 2]}
        """
        M, WW, C = feat_f0_unfold.shape

        # corner case: if no coarse matches found
        if M == 0:
            assert self.training == False, "M is always >0, when training, see coarse_matching.py"
            # logger.warning('No matches found in coarse-level.')
            data.update({
                'mkpts0_f': data['mkpts0_c'],
                'mkpts1_f': data['mkpts1_c'],
                'mconf_f': torch.zeros(0, device=feat_f0_unfold.device),
            })
            return

        # normalize
        feat_f0, feat_f1 = map(lambda feat: feat / feat.shape[-1] ** .5,
                               [feat_f0_unfold, feat_f1_unfold])
        sim_matrix = torch.einsum("nlc,nsc->nls", feat_f0,
                                  feat_f1) / self.temperature

        conf_matrix_fine = F.softmax(sim_matrix, 1) * F.softmax(sim_matrix, 2)
        data.update({'conf_matrix_fine': conf_matrix_fine})

        # predict fine-level and sub-pixel matches from conf_matrix
        data.update(**self.get_fine_match(conf_matrix_fine, feat_f0_unfold, feat_f1_unfold,
                                          feat_f0_whole, feat_f1_whole, data))

    @torch.no_grad()
    def get_fine_match(self, conf_matrix_fine, feat_f0_unfold, feat_f1_unfold, feat_f0_whole, feat_f1_whole, data):
        """
        Args:
            conf_matrix_fine (torch.Tensor): [M, WW, WW]
            feat_f0_unfold (torch.Tensor): [M, WW, C]
            feat_f1_unfold (torch.Tensor): [M, WW, C]
            feat_f0_whole (torch.Tensor): [N, C, H0_f, W0_f]
            feat_f1_whole (torch.Tensor): [N, C, H1_f, W1_f]
            data (dict)
        Update:
            data (dict):{
                'm_bids' (torch.Tensor): [M]
                'expec_f' (torch.Tensor): [M, 3],
                'mkpts0_f' (torch.Tensor): [M, 2],
                'mkpts1_f' (torch.Tensor): [M, 2]}
        """
        W_f = self.W_f

        # 1. confidence thresholding
        mask = conf_matrix_fine > self.fine_thr

        if mask.sum() == 0:
            mask[0, 0, 0] = 1
            conf_matrix_fine[0, 0, 0] = 1

        # match only the highest confidence
        mask = mask * (conf_matrix_fine == conf_matrix_fine.amax(dim=[1, 2], keepdim=True))

        # 3. find all valid fine matches
        # this only works when at most one `True` in each row
        mask_v, all_j_ids = mask.max(dim=2)
        b_ids, i_ids = torch.where(mask_v)
        j_ids = all_j_ids[b_ids, i_ids]
        mconf = conf_matrix_fine[b_ids, i_ids, j_ids]

        # 4. update with matches in original image resolution

        # indices from coarse matches
        b_ids_c, i_ids_c, j_ids_c = data['b_ids'], data['i_ids'], data['j_ids']

        # scale (coarse level / fine-level)
        scale_f_c = data['hw0_f'][0] // data['hw0_c'][0]

        # coarse level matches scaled to fine-level (1/2)
        mkpts0_c_scaled_to_f = torch.stack(
            [i_ids_c % data['hw0_c'][1], torch.div(i_ids_c, data['hw0_c'][1], rounding_mode='trunc')],
            dim=1) * scale_f_c

        mkpts1_c_scaled_to_f = torch.stack(
            [j_ids_c % data['hw1_c'][1], torch.div(j_ids_c, data['hw1_c'][1], rounding_mode='trunc')],
            dim=1) * scale_f_c

        # updated b_ids after second thresholding
        updated_b_ids = b_ids_c[b_ids]

        # scales (image res / fine level)
        scale = data['hw0_i'][0] / data['hw0_f'][0]
        scale0 = scale * data['scale0'][updated_b_ids] if 'scale0' in data else scale
        scale1 = scale * data['scale1'][updated_b_ids] if 'scale1' in data else scale

        # fine-level discrete matches on window coordiantes
        mkpts0_f_window = torch.stack(
            [i_ids % W_f, torch.div(i_ids, W_f, rounding_mode='trunc')],
            dim=1)

        mkpts1_f_window = torch.stack(
            [j_ids % W_f, torch.div(j_ids, W_f, rounding_mode='trunc')],
            dim=1)

        # unfold(crop) local windows
        mkpts0_f_unscaled = mkpts0_c_scaled_to_f[b_ids] + mkpts0_f_window - (W_f // 2)
        mkpts1_f_unscaled = mkpts1_c_scaled_to_f[b_ids] + mkpts1_f_window - (W_f // 2)

        i_ids_sub = mkpts0_f_unscaled[:, 1] * data['hw0_f'][1] + mkpts0_f_unscaled[:, 0]
        j_ids_sub = mkpts1_f_unscaled[:, 1] * data['hw1_f'][1] + mkpts1_f_unscaled[:, 0]

        feat_sub0_unfold = F.unfold(feat_f0_whole, kernel_size=3, stride=1, padding=1)
        feat_sub0_unfold = rearrange(feat_sub0_unfold, 'n (c ww) l -> n l ww c', ww=9)
        feat_sub1_unfold = F.unfold(feat_f1_whole, kernel_size=3, stride=1, padding=1)
        feat_sub1_unfold = rearrange(feat_sub1_unfold, 'n (c ww) l -> n l ww c', ww=9)

        feat_sub0_unfold = feat_sub0_unfold[updated_b_ids, i_ids_sub]
        feat_sub0_unfold = rearrange(feat_sub0_unfold, 'm (h w) c -> m c h w', h=3, w=3)
        feat_sub1_unfold = feat_sub1_unfold[updated_b_ids, j_ids_sub]
        feat_sub1_unfold = rearrange(feat_sub1_unfold, 'm (h w) c -> m c h w', h=3, w=3)

        # sub-pixel refinement
        sub_ref = self.subpixel_conv_block(torch.cat([feat_sub0_unfold, feat_sub1_unfold], dim=1)).squeeze(3).squeeze(2)
        sub_ref0, sub_ref1 = torch.chunk(sub_ref, 2, dim=-1)
        sub_ref0 = torch.tanh(sub_ref0) * 0.5
        sub_ref1 = torch.tanh(sub_ref1) * 0.5

        # final sub-pixel matches by (coarse-level + fine-level windowed + sub-pixel refinement)
        mkpts0_f_train = (mkpts0_f_unscaled + sub_ref0) * scale0
        mkpts1_f_train = (mkpts1_f_unscaled + sub_ref1) * scale1
        mkpts0_f = mkpts0_f_train.clone().detach()
        mkpts1_f = mkpts1_f_train.clone().detach()

        # These matches are the current prediction (for visualization)
        fine_matches = {
            'm_bids': b_ids_c[b_ids[mconf != 0]],  # mconf == 0 => gt matches
            'mkpts0_f': mkpts0_f[mconf != 0],
            'mkpts1_f': mkpts1_f[mconf != 0],
            'mconf_f': mconf[mconf != 0],
            'mkpts0_f_train': mkpts0_f_train[mconf != 0],
            'mkpts1_f_train': mkpts1_f_train[mconf != 0]
        }

        return fine_matches
