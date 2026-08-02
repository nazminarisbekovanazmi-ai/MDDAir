import os
import random
from PIL import Image
import numpy as np

from torch.utils.data import Dataset
from torchvision.transforms import ToPILImage, Compose, RandomCrop, ToTensor
from utils.image_utils import random_augmentation, crop_img
from utils.degradation_utils import Degradation

VALID_EXT = ('.png', '.jpg', '.jpeg', '.bmp')


def _add_gaussian_noise(clean_patch, sigma):
    noise = np.random.randn(*clean_patch.shape)
    noisy_patch = np.clip(clean_patch + noise * sigma, 0, 255).astype(np.uint8)
    return noisy_patch, clean_patch


def _list_images(directory):
    return [
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if f.lower().endswith(VALID_EXT)
    ]


class PromptTrainDataset(Dataset):
    def __init__(self, args):
        super(PromptTrainDataset, self).__init__()
        self.args = args
        self.rs_ids = []
        self.hazy_ids = []
        self.D = Degradation(args)
        self.de_type = self.args.de_type
        print(self.de_type)

        self._init_ids()
        self._merge_ids()

        self.crop_transform = Compose([
            ToPILImage(),
            RandomCrop(args.patch_size),
        ])

        self.toTensor = ToTensor()

     
    def _init_ids(self):
        if 'denoise_15' in self.de_type or 'denoise_25' in self.de_type or 'denoise_50' in self.de_type:
            self._init_clean_ids()
        if 'derain' in self.de_type:
            self._init_rs_ids()
        if 'dehaze' in self.de_type:
            self._init_hazy_ids()

        random.shuffle(self.de_type)

   
    def _init_clean_ids(self):
        clean_ids = _list_images(self.args.denoise_dir)
        if getattr(self.args, 'denoise_dir2', None):
            clean_ids += _list_images(self.args.denoise_dir2)
        if 'denoise_15' in self.de_type:
            self.s15_ids = [{"clean_id": x, "de_type": 0} for x in clean_ids] * 3
            random.shuffle(self.s15_ids)
        if 'denoise_25' in self.de_type:
            self.s25_ids = [{"clean_id": x, "de_type": 1} for x in clean_ids] * 3
            random.shuffle(self.s25_ids)
        if 'denoise_50' in self.de_type:
            self.s50_ids = [{"clean_id": x, "de_type": 2} for x in clean_ids] * 3
            random.shuffle(self.s50_ids)
        print(f"Total Denoise Ids : {len(clean_ids)}")

    def _init_hazy_ids(self):
        temp_ids = _list_images(self.args.dehaze_dir)
        self.hazy_ids = [{"clean_id": x, "de_type": 4} for x in temp_ids]
        print(f"Total Hazy Ids : {len(self.hazy_ids)}")

    def _init_rs_ids(self):
        temp_ids = _list_images(self.args.derain_dir)
        self.rs_ids = [{"clean_id": x, "de_type": 3} for x in temp_ids] * 50
        print("Total Rainy Ids : {}".format(len(self.rs_ids)))

    def _crop_patch(self, img_1, img_2):
        H = img_1.shape[0]
        W = img_1.shape[1]
        ind_H = random.randint(0, H - self.args.patch_size)
        ind_W = random.randint(0, W - self.args.patch_size)

        patch_1 = img_1[ind_H:ind_H + self.args.patch_size, ind_W:ind_W + self.args.patch_size]
        patch_2 = img_2[ind_H:ind_H + self.args.patch_size, ind_W:ind_W + self.args.patch_size]

        return patch_1, patch_2

    def _get_gt_name(self, rainy_name):
        gt_dir = os.path.join(os.path.dirname(self.args.derain_dir), 'norain')
        return os.path.join(gt_dir, os.path.basename(rainy_name).replace('x2.png', '.png'))

    def _get_nonhazy_name(self, hazy_name):
        gt_dir = os.path.join(os.path.dirname(self.args.dehaze_dir), 'GT')
        stem = os.path.splitext(os.path.basename(hazy_name))[0]
        base = stem.split('_')[0]
        for ext in ('.png', '.jpg', '.jpeg'):
            candidate = os.path.join(gt_dir, base + ext)
            if os.path.exists(candidate):
                return candidate
        return os.path.join(gt_dir, base + '.png')

    def _merge_ids(self):
        self.sample_ids = []
        if "denoise_15" in self.de_type:
            self.sample_ids += self.s15_ids
            self.sample_ids += self.s25_ids
            self.sample_ids += self.s50_ids
        if "derain" in self.de_type:
            self.sample_ids += self.rs_ids
        if "dehaze" in self.de_type:
            self.sample_ids += self.hazy_ids
        print(len(self.sample_ids))

    def __getitem__(self, idx):
        sample = self.sample_ids[idx]
        de_id = sample["de_type"]

        if de_id < 3:
            clean_id = sample["clean_id"]

            clean_img = crop_img(np.array(Image.open(clean_id).convert('RGB')), base=16)
            clean_patch = self.crop_transform(clean_img)
            clean_patch= np.array(clean_patch)

            clean_name = os.path.basename(clean_id).split('.')[0]
            
            clean_patch = random_augmentation(clean_patch)[0]

            degrad_patch = self.D.single_degrade(clean_patch, de_id)
        else:
            if de_id == 3:
                # Rain Streak Removal
                degrad_img = crop_img(np.array(Image.open(sample["clean_id"]).convert('RGB')), base=16)
                clean_name = self._get_gt_name(sample["clean_id"])
                clean_img = crop_img(np.array(Image.open(clean_name).convert('RGB')), base=16)
            elif de_id == 4:
                # Dehazing with SOTS outdoor training set
                degrad_img = crop_img(np.array(Image.open(sample["clean_id"]).convert('RGB')), base=16)
                clean_name = self._get_nonhazy_name(sample["clean_id"])
                clean_img = crop_img(np.array(Image.open(clean_name).convert('RGB')), base=16)

            degrad_patch, clean_patch = random_augmentation(*self._crop_patch(degrad_img, clean_img))

        clean_patch = self.toTensor(clean_patch)
        degrad_patch = self.toTensor(degrad_patch)
        return [clean_name, de_id], degrad_patch, clean_patch

    def __len__(self):
        return len(self.sample_ids)


# 5D 'denoise_15': 0, 'denoise_25': 1, 'denoise_50': 2, 'derain': 3, 'dehaze': 4, 'deblur' : 5 lowlight
class PromptTrainDataset5D(Dataset):
    def __init__(self, args):
        super(PromptTrainDataset5D, self).__init__()
        self.args = args
        self.rs_ids = []
        self.hazy_ids = []
        self.D = Degradation(args)
        self.de_type = self.args.de_type
        print(self.de_type)

        self._init_ids()
        self._merge_ids()

        self.crop_transform = Compose([
            ToPILImage(),
            RandomCrop(args.patch_size),
        ])

        self.toTensor = ToTensor()

    def _init_ids(self):
        if 'denoise_15' in self.de_type or 'denoise_25' in self.de_type or 'denoise_50' in self.de_type:
            self._init_clean_ids()
        if 'derain' in self.de_type:
            self._init_rs_ids()
        if 'dehaze' in self.de_type:
            self._init_hazy_ids()
        if 'deblur' in self.de_type:
            self._init_blur_ids()
        if 'lowlight' in self.de_type:
            self._init_lol_ids()

        random.shuffle(self.de_type)

    def _init_clean_ids(self):
        clean_ids = _list_images(self.args.denoise_dir)
        if getattr(self.args, 'denoise_dir2', None):
            clean_ids += _list_images(self.args.denoise_dir2)
        if 'denoise_15' in self.de_type:
            self.s15_ids = [{"clean_id": x, "de_type": 0} for x in clean_ids] * 3
            random.shuffle(self.s15_ids)
        if 'denoise_25' in self.de_type:
            self.s25_ids = [{"clean_id": x, "de_type": 1} for x in clean_ids] * 3
            random.shuffle(self.s25_ids)
        if 'denoise_50' in self.de_type:
            self.s50_ids = [{"clean_id": x, "de_type": 2} for x in clean_ids] * 3
            random.shuffle(self.s50_ids)
        print(f"Total Denoise Ids : {len(clean_ids)}")

    def _init_rs_ids(self):
        temp_ids = _list_images(self.args.derain_dir)
        self.rs_ids = [{"clean_id": x, "de_type": 3} for x in temp_ids] * 50
        print("Total Rainy Ids : {}".format(len(self.rs_ids)))

    def _init_hazy_ids(self):
        temp_ids = _list_images(self.args.dehaze_dir)
        self.hazy_ids = [{"clean_id": x, "de_type": 4} for x in temp_ids] * 2
        print(f"Total Hazy Ids : {len(self.hazy_ids)}")

    def _init_blur_ids(self):
        temp_ids = _list_images(self.args.deblur_dir)
        self.blur_ids = [{"clean_id": x, "de_type": 5} for x in temp_ids] * 10
        print(f"Total Blur Ids : {len(self.blur_ids)}")

    def _init_lol_ids(self):
        temp_ids = _list_images(self.args.lowlight_dir)
        self.lol_ids = [{"clean_id": x, "de_type": 6} for x in temp_ids] * 30
        print(f"Total LOL Ids : {len(self.lol_ids)}")

    def _crop_patch(self, img_1, img_2):
        H = img_1.shape[0]
        W = img_1.shape[1]
        ind_H = random.randint(0, H - self.args.patch_size)
        ind_W = random.randint(0, W - self.args.patch_size)

        patch_1 = img_1[ind_H:ind_H + self.args.patch_size, ind_W:ind_W + self.args.patch_size]
        patch_2 = img_2[ind_H:ind_H + self.args.patch_size, ind_W:ind_W + self.args.patch_size]

        return patch_1, patch_2

    def _get_gt_name(self, rainy_name):
        gt_dir = os.path.join(os.path.dirname(self.args.derain_dir), 'norain')
        return os.path.join(gt_dir, os.path.basename(rainy_name).replace('x2.png', '.png'))

    def _get_nonhazy_name(self, hazy_name):
        gt_dir = os.path.join(os.path.dirname(self.args.dehaze_dir), 'GT')
        stem = os.path.splitext(os.path.basename(hazy_name))[0]
        base = stem.split('_')[0]
        for ext in ('.png', '.jpg', '.jpeg'):
            candidate = os.path.join(gt_dir, base + ext)
            if os.path.exists(candidate):
                return candidate
        return os.path.join(gt_dir, base + '.png')

    def _get_sharp_name(self, blur_name):
        gt_dir = os.path.join(os.path.dirname(self.args.deblur_dir), 'sharp')
        return os.path.join(gt_dir, os.path.basename(blur_name))

    def _get_light_name(self, lol_name):
        gt_dir = os.path.join(os.path.dirname(self.args.lowlight_dir), 'high')
        return os.path.join(gt_dir, os.path.basename(lol_name))

    def _merge_ids(self):
        self.sample_ids = []
        if "denoise_15" in self.de_type:
            self.sample_ids += self.s15_ids
            self.sample_ids += self.s25_ids
            self.sample_ids += self.s50_ids
        if "derain" in self.de_type:
            self.sample_ids+= self.rs_ids
        
        if "dehaze" in self.de_type:
            self.sample_ids+= self.hazy_ids

        if "deblur" in self.de_type:
            self.sample_ids += self.blur_ids
        if "lowlight" in self.de_type:
            self.sample_ids += self.lol_ids
        
        print(len(self.sample_ids))

    def __getitem__(self, idx):
        sample = self.sample_ids[idx]
        de_id = sample["de_type"]

        if de_id < 3:
            clean_id = sample["clean_id"]

            clean_img = crop_img(np.array(Image.open(clean_id).convert('RGB')), base=16)
            clean_patch = self.crop_transform(clean_img)
            clean_patch= np.array(clean_patch)

            clean_name = os.path.basename(clean_id).split('.')[0]

            clean_patch = random_augmentation(clean_patch)[0]

            degrad_patch = self.D.single_degrade(clean_patch, de_id)
        else:
            if de_id == 3:
                # Rain Streak Removal
                degrad_img = crop_img(np.array(Image.open(sample["clean_id"]).convert('RGB')), base=16)
                clean_name = self._get_gt_name(sample["clean_id"])
                clean_img = crop_img(np.array(Image.open(clean_name).convert('RGB')), base=16)
            elif de_id == 4:
                # Dehazing with SOTS outdoor training set
                degrad_img = crop_img(np.array(Image.open(sample["clean_id"]).convert('RGB')), base=16)
                clean_name = self._get_nonhazy_name(sample["clean_id"])
                clean_img = crop_img(np.array(Image.open(clean_name).convert('RGB')), base=16)
            elif de_id == 5:
                # blur Removal
                degrad_img = crop_img(np.array(Image.open(sample["clean_id"]).convert('RGB')), base=16)
                clean_name = self._get_sharp_name(sample["clean_id"])
                clean_img = crop_img(np.array(Image.open(clean_name).convert('RGB')), base=16)
            elif de_id == 6:
                # lowlight enhancement
                degrad_img = crop_img(np.array(Image.open(sample["clean_id"]).convert('RGB')), base=16)  
                clean_name = self._get_light_name(sample["clean_id"])
                clean_img = crop_img(np.array(Image.open(clean_name).convert('RGB')), base=16)


            degrad_patch, clean_patch = random_augmentation(*self._crop_patch(degrad_img, clean_img))

        clean_patch = self.toTensor(clean_patch)
        degrad_patch = self.toTensor(degrad_patch)


        return [clean_name, de_id], degrad_patch, clean_patch

    def __len__(self):
        return len(self.sample_ids)

class DenoiseTestDataset(Dataset):
    def __init__(self, args):
        super(DenoiseTestDataset, self).__init__()
        self.args = args
        self.clean_ids = []
        self.sigma = 15

        self._init_clean_ids()

        self.toTensor = ToTensor()

    def _init_clean_ids(self):
        self.clean_ids = _list_images(self.args.denoise_path)
        if getattr(self.args, 'denoise_path2', None):
            all_ids = _list_images(self.args.denoise_path2)
            # Urban100 packages HR+LR pairs in the same folder; keep only HR ground truth
            hr_ids = [p for p in all_ids if '_HR' in os.path.basename(p)]
            self.clean_ids += hr_ids if hr_ids else all_ids
        self.num_clean = len(self.clean_ids)

    def set_sigma(self, sigma):
        self.sigma = sigma

    def __getitem__(self, clean_id):
        clean_img = crop_img(np.array(Image.open(self.clean_ids[clean_id]).convert('RGB')), base=16)
        clean_name = os.path.splitext(os.path.basename(self.clean_ids[clean_id]))[0]
        noisy_img, _ = _add_gaussian_noise(clean_img, self.sigma)
        clean_img, noisy_img = self.toTensor(clean_img), self.toTensor(noisy_img)
        return [clean_name], noisy_img, clean_img

    def __len__(self):
        return self.num_clean
    
class DerainDehazeDataset(Dataset):
    def __init__(self, args, task="derain", addnoise=False, sigma=None):
        super(DerainDehazeDataset, self).__init__()
        self.task_idx = 0
        self.args = args

        self.task_dict = {'derain': 0, 'dehaze': 1}
        self.toTensor = ToTensor()
        self.addnoise = addnoise
        self.sigma = sigma

        self.set_dataset(task)


    def _init_input_ids(self):
        _img_exts = ('.png', '.jpg', '.jpeg')
        if self.task_idx == 0:
            self.ids = [os.path.join(self.args.derain_path, f)
                        for f in os.listdir(self.args.derain_path)
                        if os.path.splitext(f)[1].lower() in _img_exts]
        else:
            self.ids = [os.path.join(self.args.dehaze_path, f)
                        for f in os.listdir(self.args.dehaze_path)
                        if os.path.splitext(f)[1].lower() in _img_exts]
        self.length = len(self.ids)

    def _get_gt_path(self, degraded_name):
        if self.task_idx == 0:
            parent = os.path.dirname(os.path.dirname(degraded_name))
            filename = os.path.basename(degraded_name)
            # Rain100L_test layout: GT_rain_test/norain-XXX.png
            c1 = os.path.join(parent, 'GT_rain_test', filename.replace('rain-', 'norain-', 1))
            if os.path.exists(c1):
                return c1
            # Legacy layout: go one more level up, norain/ sibling, strip x2 suffix
            c2 = os.path.join(os.path.dirname(parent), 'norain', filename.replace('x2.png', '.png'))
            return c2
        else:
            parent = os.path.dirname(os.path.dirname(degraded_name))
            hazy_subdir = os.path.basename(os.path.dirname(degraded_name))
            gt_subdir = (hazy_subdir.replace('hazy_outdoor', 'GT_outdoor')
                                    .replace('hazy_indoor',  'GT_indoor')
                                    .replace('hazy', 'GT'))
            filename = os.path.basename(degraded_name)
            stem = os.path.splitext(filename)[0]
            for ext in ('.jpg', '.png', '.jpeg'):
                candidate = os.path.join(parent, gt_subdir, stem + ext)
                if os.path.exists(candidate):
                    return candidate
            base = stem.split('_')[0]
            for ext in ('.jpg', '.png', '.jpeg'):
                candidate = os.path.join(parent, gt_subdir, base + ext)
                if os.path.exists(candidate):
                    return candidate
            return os.path.join(parent, gt_subdir, filename)

    def set_dataset(self, task):
        self.task_idx = self.task_dict[task]
        self._init_input_ids()

    def __getitem__(self, idx):
        degraded_path = self.ids[idx]
        clean_path = self._get_gt_path(degraded_path)

        degraded_img = crop_img(np.array(Image.open(degraded_path).convert('RGB')), base=16)
        if self.addnoise:
            degraded_img, _ = _add_gaussian_noise(degraded_img, self.sigma)
        clean_img = crop_img(np.array(Image.open(clean_path).convert('RGB')), base=16)

        clean_img, degraded_img = self.toTensor(clean_img), self.toTensor(degraded_img)
        degraded_name = os.path.splitext(os.path.basename(degraded_path))[0]

        return [degraded_name], degraded_img, clean_img

    def __len__(self):
        return self.length

#  gopro
class DeblurTestDataset(Dataset):
    def __init__(self, args, addnoise=False, sigma=None):
        super(DeblurTestDataset, self).__init__()
        self.args = args
        self.toTensor = ToTensor()
        self.addnoise = addnoise
        self.sigma = sigma
        self._init_input_ids()

    def _init_input_ids(self):
        self.ids = _list_images(os.path.join(self.args.deblur_path, 'blur'))
        self.length = len(self.ids)

    def _get_gt_path(self, degraded_name):
        parent = os.path.dirname(os.path.dirname(degraded_name))
        filename = os.path.basename(degraded_name)
        return os.path.join(parent, 'sharp', filename)

    def __getitem__(self, idx):
        degraded_path = self.ids[idx]
        clean_path = self._get_gt_path(degraded_path)

        degraded_img = crop_img(np.array(Image.open(degraded_path).convert('RGB')), base=16)
        if self.addnoise:
            degraded_img, _ = _add_gaussian_noise(degraded_img, self.sigma)
        clean_img = crop_img(np.array(Image.open(clean_path).convert('RGB')), base=16)

        clean_img, degraded_img = self.toTensor(clean_img), self.toTensor(degraded_img)
        degraded_name = os.path.splitext(os.path.basename(degraded_path))[0]

        return [degraded_name], degraded_img, clean_img

    def __len__(self):
        return self.length

    
#  LOL
class LOLTestDataset(Dataset):
    def __init__(self, args, addnoise=False, sigma=None):
        super(LOLTestDataset, self).__init__()
        self.args = args
        self.toTensor = ToTensor()
        self.addnoise = addnoise
        self.sigma = sigma
        self._init_input_ids()

    def _init_input_ids(self):
        self.ids = _list_images(os.path.join(self.args.lowlight_path, 'low'))
        self.length = len(self.ids)

    def _get_gt_path(self, degraded_name):
        normalized = degraded_name.replace('\\', '/')
        return normalized.replace('/low/', '/high/')

    def __getitem__(self, idx):
        degraded_path = self.ids[idx]
        clean_path = self._get_gt_path(degraded_path)

        degraded_img = crop_img(np.array(Image.open(degraded_path).convert('RGB')), base=16)
        if self.addnoise:
            degraded_img, _ = _add_gaussian_noise(degraded_img, self.sigma)
        clean_img = crop_img(np.array(Image.open(clean_path).convert('RGB')), base=16)

        clean_img, degraded_img = self.toTensor(clean_img), self.toTensor(degraded_img)
        degraded_name = os.path.splitext(os.path.basename(degraded_path))[0]

        return [degraded_name], degraded_img, clean_img

    def __len__(self):
        return self.length




