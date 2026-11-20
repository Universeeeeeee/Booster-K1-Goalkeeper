from pathlib import Path

import cv2
import numpy as np

try:
    import torch
    import torch.nn.functional as F
except ModuleNotFoundError:
    torch = None
    F = None


SCRIPT_DIR = Path(__file__).resolve().parent
RAW_PHOTO_DIR = SCRIPT_DIR / "raw_photo"
PY_GOLDEN_DIR = SCRIPT_DIR / "py_golden"
THRESHOLD = 3


def compute_tdsd(curr_tensor, last_tensor):
    """
    使用 PyTorch 计算当前帧的空间差分和相邻帧的时间差分。
    grad_up    = curr[y, x] - curr[y - 1, x]
    grad_right = curr[y, x] - curr[y, x + 1]
    grad_td    = curr - last
    """
    if torch is None:
        raise RuntimeError("当前环境未安装 torch，请使用 compute_tdsd_numpy 或安装 torch。")

    if len(curr_tensor.shape) == 3:
        curr_tensor = curr_tensor.unsqueeze(0)
    elif len(curr_tensor.shape) == 2:
        curr_tensor = curr_tensor.unsqueeze(0).unsqueeze(0)

    if last_tensor is None:
        grad_td = None
    else:
        if len(last_tensor.shape) == 3:
            last_tensor = last_tensor.unsqueeze(0)
        elif len(last_tensor.shape) == 2:
            last_tensor = last_tensor.unsqueeze(0).unsqueeze(0)
        grad_td = curr_tensor - last_tensor

    _, channels, _, _ = curr_tensor.shape

    kernel_up = torch.tensor([[[[0, -1, 0],
                                [0, 1, 0],
                                [0, 0, 0]]]], dtype=torch.float32)
    kernel_right = torch.tensor([[[[0, 0, 0],
                                   [0, 1, -1],
                                   [0, 0, 0]]]], dtype=torch.float32)

    if channels > 1:
        kernel_up = kernel_up.repeat(1, channels, 1, 1)
        kernel_right = kernel_right.repeat(1, channels, 1, 1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    curr_tensor = curr_tensor.to(device)
    kernel_up = kernel_up.to(device)
    kernel_right = kernel_right.to(device)

    grad_up = F.conv2d(curr_tensor, kernel_up, groups=channels, padding=1)
    grad_right = F.conv2d(curr_tensor, kernel_right, groups=channels, padding=1)
    grad_up[:, :, 0, :] = 0
    grad_right[:, :, :, -1] = 0

    return torch.trunc(grad_up / 2), torch.trunc(grad_right / 2), grad_td


def compute_tdsd_numpy(curr_image, last_image=None, *, legacy_sd=False):
    """SD 除以 2 后向零截断。legacy_sd 用于复核历史录制。"""
    curr = curr_image.astype(np.int16)

    grad_up = np.zeros_like(curr, dtype=np.int16)
    grad_up[1:, :] = curr[1:, :] - curr[:-1, :]

    grad_right = np.zeros_like(curr, dtype=np.int16)
    grad_right[:, :-1] = curr[:, :-1] - curr[:, 1:]

    grad_td = None
    if last_image is not None:
        grad_td = curr - last_image.astype(np.int16)

    if not legacy_sd:
        grad_up = np.trunc(grad_up / 2).astype(np.int16)
        grad_right = np.trunc(grad_right / 2).astype(np.int16)
    return grad_up, grad_right, grad_td


def read_gray_image(path):
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise RuntimeError(f"读取灰度图失败: {path}")
    return image


def diff_to_u8(diff):
    return np.clip(np.abs(diff), 0, 255).astype(np.uint8)


def apply_threshold(image, threshold):
    thresholded = image.copy()
    thresholded[thresholded <= threshold] = 0
    return thresholded


def write_image(path, image):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise OSError(f"写入图片失败: {path}")


def generate_requested_outputs():
    photo_1_path = RAW_PHOTO_DIR / "photo_1.bmp"
    photo_2_path = RAW_PHOTO_DIR / "photo_2.bmp"
    photo_1 = read_gray_image(photo_1_path)
    photo_2 = read_gray_image(photo_2_path)

    grad_up, grad_right, _ = compute_tdsd_numpy(photo_1)
    sd_up_image = diff_to_u8(grad_up)
    sd_right_image = diff_to_u8(grad_right)
    sd_up_path = SCRIPT_DIR / "photo_1_sd_up.png"
    sd_right_path = SCRIPT_DIR / "photo_1_sd_right.png"
    write_image(sd_up_path, sd_up_image)
    write_image(sd_right_path, sd_right_image)

    sd_up_th_path = PY_GOLDEN_DIR / f"photo_1_sd_up_th{THRESHOLD}.png"
    sd_right_th_path = PY_GOLDEN_DIR / f"photo_1_sd_right_th{THRESHOLD}.png"
    write_image(sd_up_th_path, apply_threshold(sd_up_image, THRESHOLD))
    write_image(sd_right_th_path, apply_threshold(sd_right_image, THRESHOLD))

    _, _, grad_td = compute_tdsd_numpy(photo_2, photo_1)
    td_image = diff_to_u8(grad_td)
    td_path = SCRIPT_DIR / "photo_1_2_td.png"
    write_image(td_path, td_image)

    td_th_path = PY_GOLDEN_DIR / f"photo_1_2_td_th{THRESHOLD}.png"
    write_image(td_th_path, apply_threshold(td_image, THRESHOLD))

    print(f"SD input : {photo_1_path.name}")
    print(f"SD_UP output   : {sd_up_path.name}")
    print(f"SD_RIGHT output: {sd_right_path.name}")
    print(f"SD_UP th{THRESHOLD} output   : {sd_up_th_path.relative_to(SCRIPT_DIR)}")
    print(f"SD_RIGHT th{THRESHOLD} output: {sd_right_th_path.relative_to(SCRIPT_DIR)}")
    print(f"TD input : {photo_1_path.name}, {photo_2_path.name}")
    print(f"TD output: {td_path.name}")
    print(f"TD th{THRESHOLD} output: {td_th_path.relative_to(SCRIPT_DIR)}")


if __name__ == "__main__":
    generate_requested_outputs()
