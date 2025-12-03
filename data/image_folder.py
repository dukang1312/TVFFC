###############################################################################
# Code from
# https://github.com/pytorch/vision/blob/master/torchvision/datasets/folder.py
# Modified the original code so that it also loads images from the current
# directory as well as the subdirectories
###############################################################################
import torch.utils.data as data
from PIL import Image
import os
import glob
import numpy as np

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.tiff','.tif'
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir):
    dir1 = dir+'/*.tiff'
    dir2 = dir +'/*.tif'
    dir3 = dir +"/*.png"
    images1 = glob.glob(dir1)
    images2 = glob.glob(dir2)
    images3 = glob.glob(dir3)
    images2.extend(images1)
    images2.extend(images3)


    # print(dir)
    # images = []
    # assert os.path.isdir(dir), '%s is not a valid directory' % dir
    #
    #
    # for root, _, fnames in sorted(os.walk(dir)):
    #     for fname in fnames:
    #         if is_image_file(fname):
    #             path = os.path.join(root, fname)
    #             images.append(path)
    #             print(images)

    return images2


def default_loader(image):

    real_img=np.asarray(image)


    real_img=(real_img/65535.-0.5)*2


    size=real_img.shape


    a,b=size


    image=np.zeros([a,b,3])
    image[:,:,0]=real_img
    image[:,:,1] = real_img
    image[:,:,2] = real_img

    return image,size




class ImageFolder(data.Dataset):

    def __init__(self, root, transform=None, return_paths=False,
                 loader=default_loader):
        imgs = make_dataset(root)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in: " + root + "\n"
                               "Supported image extensions are: " +
                               ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.imgs = imgs
        self.transform = transform
        self.return_paths = return_paths
        self.loader = loader

    def __getitem__(self, index):
        path = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.return_paths:
            return img, path
        else:
            return img

    def __len__(self):
        return len(self.imgs)
