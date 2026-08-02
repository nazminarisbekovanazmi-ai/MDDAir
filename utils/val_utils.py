import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


class AverageMeter():
    """ Computes and stores the average and current value """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def compute_psnr_ssim(recoverd, clean):
    assert recoverd.shape == clean.shape
    recoverd = np.clip(recoverd.detach().cpu().numpy(), 0, 1).transpose(0, 2, 3, 1)
    clean = np.clip(clean.detach().cpu().numpy(), 0, 1).transpose(0, 2, 3, 1)
    psnr = 0
    ssim = 0
    for i in range(recoverd.shape[0]):
        psnr += peak_signal_noise_ratio(clean[i], recoverd[i], data_range=1)
        ssim += structural_similarity(clean[i], recoverd[i], data_range=1, channel_axis=-1)  # type: ignore[operator]
    return psnr / recoverd.shape[0], ssim / recoverd.shape[0], recoverd.shape[0]
