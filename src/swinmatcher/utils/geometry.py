import torch


@torch.no_grad()
def warp_kpts(kpts0, T_0to1):
    """
    Args:
        kpts0 (torch.Tensor): (N, L, 2),
        T_0to1 (torch.Tensor): (N, 3, 3)
    Returns:
        valid_mask (torch.Tensor): (N, L),
        warped_kpts0 (torch.Tensor): (N, L, 2)
    """
    kpts0_homo = torch.cat((kpts0.permute(0, 2, 1), torch.ones(kpts0.shape[0], 1, kpts0.shape[1],
                                                               device=kpts0.device)), dim=1)  # (N, 3, L)
    warped_kpts0_homo = torch.matmul(T_0to1, kpts0_homo)  # (N, 3, L)
    warped_kpts0 = (warped_kpts0_homo[:, :2, :] / warped_kpts0_homo[:, 2:, :]).permute(0, 2, 1)  # (N, L, 2)
    w, h = 512, 512
    valid_mask = (warped_kpts0[:, :, 0] >= 0) & (warped_kpts0[:, :, 0] <= w - 1) & \
                 (warped_kpts0[:, :, 1] >= 0) & (warped_kpts0[:, :, 1] <= h - 1)  # (N, L)
    return valid_mask, warped_kpts0


@torch.no_grad()
def warp_kpts_fine(kpts0, T_0to1, b_ids):
    """
    Args:
        kpts0 (torch.Tensor): (N, L, 2),
        T_0to1 (torch.Tensor): (N, 3, 3)
        b_ids (torch.Tensor): [M], selected batch ids for fine-level matching
    Returns:
        valid_mask (torch.Tensor): (N, L),
        warped_kpts0 (torch.Tensor): (N, L, 2)
    """
    kpts0_homo = torch.cat((kpts0.permute(0, 2, 1), torch.ones(kpts0.shape[0], 1, kpts0.shape[1],
                                                               device=kpts0.device)), dim=1)  # (N, 3, L)
    warped_kpts0_homo = torch.matmul(T_0to1[b_ids], kpts0_homo)  # (N, 3, L)
    warped_kpts0 = (warped_kpts0_homo[:, :2, :] / warped_kpts0_homo[:, 2:, :]).permute(0, 2, 1)  # (N, L, 2)
    w, h = 512, 512
    valid_mask = (warped_kpts0[:, :, 0] >= 0) & (warped_kpts0[:, :, 0] <= w - 1) & \
                 (warped_kpts0[:, :, 1] >= 0) & (warped_kpts0[:, :, 1] <= h - 1)  # (N, L)
    return valid_mask, warped_kpts0
