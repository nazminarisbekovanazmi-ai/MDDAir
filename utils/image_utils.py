import random
import numpy as np


# crop an image to the nearest multiple of base
def crop_img(image, base=64):
    h = image.shape[0]
    w = image.shape[1]
    crop_h = h % base
    crop_w = w % base
    return image[crop_h // 2:h - crop_h + crop_h // 2, crop_w // 2:w - crop_w + crop_w // 2, :]


def data_augmentation(image, mode):
    if mode == 1:
        out = np.flipud(image)
    elif mode == 2:
        out = np.rot90(image)
    elif mode == 3:
        out = np.flipud(np.rot90(image))
    elif mode == 4:
        out = np.rot90(image, k=2)
    elif mode == 5:
        out = np.flipud(np.rot90(image, k=2))
    elif mode == 6:
        out = np.rot90(image, k=3)
    elif mode == 7:
        out = np.flipud(np.rot90(image, k=3))
    else:
        raise ValueError(f'Invalid augmentation mode: {mode}')
    return out


def random_augmentation(*args):
    flag_aug = random.randint(1, 7)
    return [data_augmentation(data, flag_aug).copy() for data in args]
