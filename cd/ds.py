from torch.utils.data import Dataset
import torch
from PIL import Image
from pathlib import Path
import numpy as np
from dataclasses import dataclass
import random

@dataclass
class imageset:
    t1: Path
    t2: Path
    cm: Path


@dataclass
class patch:
    imset: imageset
    x: tuple
    y: tuple


class CDDataset(Dataset):
    """"""

    imagesets = None
    patchsize = None
    nx = 0
    ny = 0
    patches = []
    normalize = True

    cache = {}
    def loadrgb(self, image):
      if str(image) not in self.cache:
        img = self._loadrgb(image)
        #if self.normalize:
        #  img = (img-img.mean(axis=(-1,-2))[:,None,None])/img.std(axis=(-1,-2))[:,None,None]
        self.cache[str(image)] = img
      return self.cache[str(image)]  
    
    def loadcm(self, image):
      if str(image) not in self.cache:
        self.cache[str(image)]=self._loadcm(image)
      return self.cache[str(image)]

    def __init__(self):
        if self.imagesets is None or self.patchsize is None:
            raise NotImplementedError
        m, v = np.zeros(3), np.zeros(3)
        self.patches = []
        for imset in self.imagesets:
            im1 = self.loadrgb(imset.t1)
            im2 = self.loadrgb(imset.t2)
            cm = self.loadcm(imset.cm)
            assert im1.shape[1:] == im2.shape[1:] == cm.shape
            assert im1.shape[0] == im2.shape[0] == 3    
            for ix in range(im1.shape[1] // self.patchsize):
                for iy in range(im1.shape[2] // self.patchsize):
                    self.patches.append(
                        patch(
                            imset,
                            (self.patchsize * ix, self.patchsize * (ix + 1)),
                            (self.patchsize * iy, self.patchsize * (iy + 1)),
                        )
                    )
            self.nx += ix / len(self.imagesets)
            self.ny += iy / len(self.imagesets)
        self._m=m
        self._s=np.sqrt(v)
    def __getitem__(self, idx):
        patch = self.patches[idx]
        im1 = self.loadrgb(patch.imset.t1).astype(np.float64)
        im2 = self.loadrgb(patch.imset.t2).astype(np.float64)
        cm = self.loadcm(patch.imset.cm).astype(bool)
        im1 = im1[..., patch.x[0] : patch.x[1], patch.y[0] : patch.y[1]]
        im2 = im2[..., patch.x[0] : patch.x[1], patch.y[0] : patch.y[1]]
        if self.normalize:
          im1=(im1-im1.mean(axis=(-1,-2))[:,None,None])/im1.std(axis=(-1,-2))[:,None,None]
          im2=(im2-im2.mean(axis=(-1,-2))[:,None,None])/im2.std(axis=(-1,-2))[:,None,None]
        cm = cm[..., patch.x[0] : patch.x[1], patch.y[0] : patch.y[1]]
        return (im1, im2, cm)

    def __len__(self):
        return len(self.patches)


class WV_S1(CDDataset):
    def __init__(self, path: Path, patchsize: int):
        self.imagesets = [imageset(*(path / f for f in ["t1.bmp", "t2.bmp", "gt.bmp"]))]
        self.patchsize = patchsize
        super(WV_S1, self).__init__()

    def _loadrgb(self, image):
        return np.array(Image.open(image)).transpose(2, 0, 1) / 255

    def _loadcm(self, image):
        return np.array(Image.open(image)) < 128


class OSCD(CDDataset):
    def __init__(self, path: Path, patchsize: int):

        self.imagesets = [
            imageset(im1, im2, cm)
            for im1, im2, cm in zip(
                sorted((path / "images").rglob("imgs_1_rect")),
                sorted((path / "images").rglob("imgs_2_rect")),
                sorted((path / "labels").rglob("cm")),
            )
        ]
        self.patchsize = patchsize
        super(OSCD, self).__init__()

    def _loadrgb(self, image):
        return np.stack([np.array(Image.open(image / b)) for b in ("B02.tif", "B03.tif", "B04.tif")])
        
    def _loadcm(self, image):
        return np.array(Image.open(next(image.glob("*-cm.tif"))))>1


from typing import Tuple
from torch.utils.data import Subset


def split(
    ds: Dataset, validation_ratio: float, test_ratio: float, runsize=16, seed=0
) -> Tuple[Dataset, Dataset, Dataset]:
    """
    splits dataset by ratio (0..1) of validation and test in validation, test and train (remainder)
    while ensuring somewhat equal distribution between different parts
    of the Dataset by randomly choosing out of partitions of size runsize
    """
    rng = np.random.RandomState(0)
    val = list()
    test = list()
    train = list()
    split = np.array_split(np.arange(len(ds)), len(ds) / runsize)
    for s in split:
        nv = int(validation_ratio * (len(val) + len(test) + len(train) + len(s)) - len(val))
        i = rng.choice(s, nv, replace=False)
        s = np.setdiff1d(s, i)
        val += i.tolist()
        nt = int(test_ratio * (len(val) + len(test) + len(train) + len(s)) - len(test))
        i = rng.choice(s, nt, replace=False)
        s = np.setdiff1d(s, i)
        test += i.tolist()
        train += s.tolist()
    return CDSubset(ds, train), CDSubset(ds, val), CDSubset(ds, test)


class CDSubset(Subset):
    """
    Subset of a CDDataset at specified indices with optional augmentation.
    """

    augment = False

    def __getitem__(self, idx):
        im1, im2, cm = super().__getitem__(idx)
        if self.augment:
            if random.randint(0, 1):
                im1 = np.swapaxes(im1, -1, -2)
                im2 = np.swapaxes(im2, -1, -2)
                cm = np.swapaxes(cm, -1, -2)
            rot = random.randint(0, 3)
            im1 = np.copy(np.rot90(im1, rot, (-1, -2)))
            im2 = np.copy(np.rot90(im2, rot, (-1, -2)))
            cm = np.copy(np.rot90(cm, rot, (-1, -2)))

        return im1, im2, cm
