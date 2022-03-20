import os
import os.path as path
import torch
import numpy as np
import scipy.io as sio
from torch.utils.data import Dataset
from .misc import read_dir

def get_dataset(
    dataset_dir, dataset_name, image_size=None, opts=None):
    if dataset_name in {"aapm_sparse", "aapm_learn", "aapm_official"}:
        return AAPM(
            dataset_dir, dataset_name, image_size, opts.residual)
    else:
        raise ValueError("datasets.get_dataset: invalid dataset name.")


def random_crop(image, crop_size=None):
    """ Random crop an image

    Args:
        image: an image to be cropped
        crop_size: the size of cropped image
    """
    if crop_size is None:
        return image

    if np.isscalar(crop_size): crop_size = (crop_size, crop_size)

    assert len(crop_size) == 2 and \
        np.all(np.less_equal(crop_size, image.shape[:2])), \
        "random_crop: invalid image size"

    crop_range = np.array(image.shape[:2]) - crop_size
    crop_x = np.random.randint(crop_range[0] + 1)
    crop_y = np.random.randint(crop_range[1] + 1)

    return image[crop_x:crop_x + crop_size[0],
        crop_y:crop_y + crop_size[1], ...]


class AAPM(torch.utils.data.Dataset):
    def __init__(self, dataset_dir='data/aapm_sparse/train',
        dataset_name="aapm_sparse", image_size=256, residual=False,
        with_sinogram=False):
        super(AAPM, self).__init__()

        study_dirs = read_dir(dataset_dir, 'dir')
        self.data_files = [f for d in study_dirs
            for f in read_dir(d, lambda x: x.endswith('mat'))]

        if np.isscalar(image_size): image_size = (image_size, image_size)
        self.image_size = image_size
        self.dataset_name = dataset_name
        self.residual = residual
        self.with_sinogram = with_sinogram

    def to_tensor(self, data, norm=True):
        if data.ndim == 2:
            data = data[np.newaxis, ...]
        if norm:
            data = self.normalize(data)
        data = data * 2.0 - 1.0
        data = torch.FloatTensor(data)

        return data

    def to_numpy(self, data, denorm=True):
        data = data.detach().cpu().numpy()
        data = data * 0.5 + 0.5
        if denorm:
            data = self.denormalize(data)
        return data

    def load_data(self, data_file):
        data = sio.loadmat(data_file)
        if self.dataset_name in {'aapm_sparse'}:
            if self.with_sinogram:
                return (data['dense_view'], data['sparse_view'],
                    data['dense_sinogram'], data['sparse_sinogram'])
            else:
                return data['dense_view'], data['sparse_view']
        return data

    def normalize(self, data):
        # (-0.031210732, 0.088769846)
        if self.dataset_name in {'aapm_sparse'}:
            data_min = -0.035
            data_max = 0.09
            data = (data - data_min) / (data_max - data_min)
        return data

    def denormalize(self, data):
        if self.dataset_name in {'aapm_sparse'}:
            data_min = -0.035
            data_max = 0.09
            data = data * (data_max - data_min) + data_min
        return data

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, index):
        data_file = self.data_files[index]
        data_name = path.basename(data_file)

        # load images
        data = self.load_data(data_file)
        crops = random_crop(np.stack(data[:2], -1), self.image_size)
        hq_image, lq_image = crops[..., 0], crops[..., 1]

        hq_image = self.to_tensor(hq_image)
        lq_image = self.to_tensor(lq_image)
        if self.residual:
            hq_image -= lq_image
            hq_image *= 0.5

        if self.with_sinogram:
            hq_sinogram = self.to_tensor(data[2], True)
            lq_sinogram = self.to_tensor(data[3], True)
            if self.residual: hq_sinogram -= lq_sinogram

            return {"data_name": data_name,
                "hq_image": hq_image,  "lq_image": lq_image,
                "hq_sinogram": hq_sinogram, "lq_sinogram": lq_sinogram}
        else:
            return {"data_name": data_name,
                "hq_image": hq_image, "lq_image": lq_image}
