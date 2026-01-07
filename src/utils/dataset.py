import torch
# import io
import cv2
# import h5py
import random
import numpy as np
import albumentations as A


try:
    # for internel use only
    from .client import MEGADEPTH_CLIENT, SCANNET_CLIENT
except Exception:
    MEGADEPTH_CLIENT = SCANNET_CLIENT = None


# --- DATA IO ---

# def load_array_from_s3(
#     path, client, cv_type,
#     use_h5py=False,
# ):
#     byte_str = client.Get(path)
#     try:
#         if not use_h5py:
#             raw_array = np.fromstring(byte_str, np.uint8)
#             data = cv2.imdecode(raw_array, cv_type)
#         else:
#             f = io.BytesIO(byte_str)
#             data = np.array(h5py.File(f, 'r')['/depth'])
#     except Exception as ex:
#         print(f"==> Data loading failure: {path}")
#         raise ex
#
#     assert data is not None
#     return data


def imread_gray(path, augment_fn=None, client=SCANNET_CLIENT, to_thermal=False):
    # cv_type = cv2.IMREAD_GRAYSCALE if augment_fn is None \
    #             else cv2.IMREAD_COLOR
    # if str(path).startswith('s3://'):
    #     image = load_array_from_s3(str(path), client, cv_type)
    # else:
    #     image = cv2.imread(str(path), cv_type)
    if to_thermal:
        image = cv2.imread(path)
        transform = RGBtoThermal()
        image = transform.augment_pseudo_thermal(image)
    else:
        # image = cv2.imread(path, 0)
        image = cv2.imread(path)
        choice = random.choices([0, 1, 2], [1 / 3, 2 / 9, 4 / 9])[0]
        if choice == 0:
            gamma_value = 1
        elif choice == 1:
            gamma_value = random.uniform(0.3, 0.6)
        else:
            gamma_value = random.uniform(2, 4)
        image = adjust_gamma(image, gamma=gamma_value)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # if augment_fn is not None:
    #     image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    #     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #     image = augment_fn(image)
    #     image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return image  # (h, w)


def get_resized_wh(w, h, resize=None):
    if resize is not None:  # resize the longer edge
        scale = resize / max(h, w)
        w_new, h_new = int(round(w*scale)), int(round(h*scale))
    else:
        w_new, h_new = w, h
    return w_new, h_new


def get_divisible_wh(w, h, df=None):
    if df is not None:
        w_new, h_new = map(lambda x: int(x // df * df), [w, h])
    else:
        w_new, h_new = w, h
    return w_new, h_new


def pad_bottom_right(inp, pad_size, ret_mask=False):
    assert isinstance(pad_size, int) and pad_size >= max(inp.shape[-2:]), f"{pad_size} < {max(inp.shape[-2:])}"
    mask = None
    if inp.ndim == 2:
        padded = np.zeros((pad_size, pad_size), dtype=inp.dtype)
        padded[:inp.shape[0], :inp.shape[1]] = inp
        if ret_mask:
            mask = np.zeros((pad_size, pad_size), dtype=bool)
            mask[:inp.shape[0], :inp.shape[1]] = True
    elif inp.ndim == 3:
        padded = np.zeros((inp.shape[0], pad_size, pad_size), dtype=inp.dtype)
        padded[:, :inp.shape[1], :inp.shape[2]] = inp
        if ret_mask:
            mask = np.zeros((inp.shape[0], pad_size, pad_size), dtype=bool)
            mask[:, :inp.shape[1], :inp.shape[2]] = True
    else:
        raise NotImplementedError()
    return padded, mask


def read_multi_modality_gray(path, resize=None, df=None, padding=False, augment_fn=None, to_thermal=False):
    """
    Args:
        resize (int, optional): the longer edge of resized images. None for no resize.
        padding (bool): If set to 'True', zero-pad resized images to squared size.
        augment_fn (callable, optional): augments images with pre-defined visual effects
    Returns:
        image (torch.tensor): (1, h, w)
        mask (torch.tensor): (h, w)
        scale (torch.tensor): [w/w_new, h/h_new]
    """
    # read image
    image = imread_gray(path, augment_fn, client=MEGADEPTH_CLIENT, to_thermal=to_thermal)

    # resize image
    w, h = image.shape[1], image.shape[0]
    w_new, h_new = get_resized_wh(w, h, resize)
    w_new, h_new = get_divisible_wh(w_new, h_new, df)

    image = cv2.resize(image, (w_new, h_new))
    scale = torch.tensor([w/w_new, h/h_new], dtype=torch.float)

    if padding:  # padding
        pad_to = max(h_new, w_new)
        image, mask = pad_bottom_right(image, pad_to, ret_mask=True)
    else:
        mask = None

    image = torch.from_numpy(image).float()[None] / 255  # (h, w) -> (1, h, w) and normalized
    mask = torch.from_numpy(mask)

    return image, mask, scale


class RGBtoThermal:
    def __init__(self):
        self.blur = A.Blur(p=0.7, blur_limit=(3, 5))  # default: blur_limit=(2, 4)
        self.hsv = A.HueSaturationValue(p=0.9, val_shift_limit=(-30, +30), hue_shift_limit=(-90, +90),
                                        sat_shift_limit=(-30, +30))

        # parameters for the cosine transform
        self.w_0 = np.pi * 2 / 3
        self.w_r = np.pi / 2
        self.theta_r = np.pi / 2

    def augment_pseudo_thermal(self, image):
        # HSV augmentation
        image = self.hsv(image=image)["image"]

        # Random blur
        image = self.blur(image=image)["image"]

        # Convert the image to the gray scale
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Normalize the image between (-0.5, 0.5)
        image = image / 255 - 0.5

        # Random phase and freq for the cosine transform
        phase = np.pi / 2 + np.random.randn(1) * self.theta_r
        w = self.w_0 + np.abs(np.random.randn(1)) * self.w_r

        # Cosine transform
        image = np.cos(image * w + phase)

        # Min-max normalization for the transformed image
        image = (image - image.min()) / (image.max() - image.min()) * 255

        return image.astype(np.uint8)


def adjust_gamma(image, gamma=1.0):
    inv_gamma = 1.0 / gamma
    return ((image / 255) ** inv_gamma * 255).astype("uint8")
