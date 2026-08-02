import random
import numpy as np


class Degradation(object):
    def __init__(self, args):
        super(Degradation, self).__init__()
        self.args = args

    def _add_gaussian_noise(self, clean_patch, sigma):
        noise = np.random.randn(*clean_patch.shape)
        noisy_patch = np.clip(clean_patch + noise * sigma, 0, 255).astype(np.uint8)
        return noisy_patch, clean_patch

    def _degrade_by_type(self, clean_patch, degrade_type):
        if degrade_type == 0:
            degraded_patch, clean_patch = self._add_gaussian_noise(clean_patch, sigma=15)
        elif degrade_type == 1:
            degraded_patch, clean_patch = self._add_gaussian_noise(clean_patch, sigma=25)
        elif degrade_type == 2:
            degraded_patch, clean_patch = self._add_gaussian_noise(clean_patch, sigma=50)
        return degraded_patch, clean_patch

    def single_degrade(self, clean_patch, degrade_type=None):
        if degrade_type is None:
            degrade_type = random.randint(0, 2)
        degrad_patch, _ = self._degrade_by_type(clean_patch, degrade_type)
        return degrad_patch
