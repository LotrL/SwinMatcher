from .swin_unet import SwinTransformerSys


def build_backbone():
    return SwinTransformerSys(img_size=512, patch_size=4, in_chans=1, embed_dim=96, depths=[2, 2, 2, 2],
                              num_heads=[4, 8, 16, 32], window_size=16, mlp_ratio=4.0, qkv_bias=True, qk_scale=None,
                              drop_rate=0.0, attn_drop_rate=0.2, drop_path_rate=0.3, ape=False, patch_norm=True,
                              use_checkpoint=False)
