import torch
import os
import cv2
import warnings
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from src.swinmatcher import SwinMatcher, default_cfg

warnings.filterwarnings("ignore")


def draw_matches(image0, image1, points0, points1):
    difference = image0.shape[0] - image1.shape[0]
    if difference < 0:
        top = abs(difference) // 2
        bottom = abs(difference) - top
        image0 = cv2.copyMakeBorder(image0, top, bottom, 0, 0, cv2.BORDER_CONSTANT)
        if len(points0) > 0:
            points0[:, 1] += top
    elif difference > 0:
        top = difference // 2
        bottom = difference - top
        image1 = cv2.copyMakeBorder(image1, top, bottom, 0, 0, cv2.BORDER_CONSTANT)
        if len(points1) > 0:
            points1[:, 1] += top
    points0 = [cv2.KeyPoint(point0[0], point0[1], 1) for point0 in points0]
    points1 = [cv2.KeyPoint(point1[0], point1[1], 1) for point1 in points1]
    matches = [cv2.DMatch(index, index, 1) for index in range(len(points0))]
    # noinspection PyTypeChecker
    matches_visual = cv2.drawMatches(image0, points0, image1, points1, matches, None, (0, 255, 0))
    return matches_visual


if __name__ == "__main__":
    # If matching is difficult, you can try lowering the threshold.
    config = default_cfg
    config["match_coarse"]["thr"] = 0.3  # default: 0.3
    config["match_coarse"]["border_rm"] = 2  # default: 2
    config["fine"]["thr"] = 0.1  # default: 0.1
    config["img_size"] = 512  # default: 512

    half_precision = True  # default: False

    matcher = SwinMatcher(config=config)
    matcher.load_state_dict(torch.load("weights/swinmatcher.ckpt")["state_dict"])
    matcher = matcher.eval().cuda()
    if half_precision:
        matcher = matcher.half()

    folder_path = "sample"
    image_names = os.listdir(folder_path)
    for i in tqdm(range(0, len(image_names), 2)):
        image0_bgr = cv2.imread(folder_path + "/" + image_names[i])
        image1_bgr = cv2.imread(folder_path + "/" + image_names[i + 1])
        image0_rgb = cv2.cvtColor(image0_bgr, cv2.COLOR_BGR2RGB)
        image1_rgb = cv2.cvtColor(image1_bgr, cv2.COLOR_BGR2RGB)
        image0_gray = cv2.cvtColor(image0_bgr, cv2.COLOR_BGR2GRAY)
        image1_gray = cv2.cvtColor(image1_bgr, cv2.COLOR_BGR2GRAY)

        height0, width0 = image0_gray.shape
        new_width0, new_height0 = config["img_size"], config["img_size"]
        image0_gray = cv2.resize(image0_gray, (new_width0, new_height0))
        scale0 = np.array([width0 / new_width0, height0 / new_height0], dtype=np.float16)

        height1, width1 = image1_gray.shape
        new_width1, new_height1 = config["img_size"], config["img_size"]
        image1_gray = cv2.resize(image1_gray, (new_width1, new_height1))
        scale1 = np.array([width1 / new_width1, height1 / new_height1], dtype=np.float16)

        image0 = torch.from_numpy(image0_gray)[None][None].cuda() / 255
        image1 = torch.from_numpy(image1_gray)[None][None].cuda() / 255
        if half_precision:
            image0 = image0.half()
            image1 = image1.half()
        batch = {"image0": image0, "image1": image1}

        with torch.no_grad():
            matcher(batch)
            matched_points0 = batch["mkpts0_f"].cpu().numpy() * scale0
            matched_points1 = batch["mkpts1_f"].cpu().numpy() * scale1
        torch.cuda.empty_cache()

        if len(matched_points0) >= 4:
            homography, mask = cv2.findHomography(matched_points1, matched_points0, cv2.USAC_MAGSAC, 5.0)
            matched_points0 = matched_points0[mask.flatten() == 1]
            matched_points1 = matched_points1[mask.flatten() == 1]

        matches_visual = draw_matches(image0_rgb, image1_rgb, matched_points0, matched_points1)

        plt.figure()
        plt.imshow(matches_visual, "gray")
        plt.title(f"{len(matched_points0)} matches")
        plt.axis("off")

    plt.show()
