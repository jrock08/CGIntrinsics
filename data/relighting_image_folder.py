import torch.utils.data as data
import torch

from PIL import Image
import numpy as np
from skimage.transform import resize

import os
import os.path
import sys

IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', 'webp']


def has_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.

    Args:
        filename (string): path to a file
        extensions (iterable of strings): extensions to consider (lowercase)

    Returns:
        bool: True if the filename ends with one of given extensions
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)


def is_image_file(filename):
    """Checks if a file is an allowed image extension.

    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    return has_file_allowed_extension(filename, IMG_EXTENSIONS)

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)
 
def make_dataset(dir, class_to_idx, extensions):
    images = []
    dir = os.path.expanduser(dir)
    for target in sorted(class_to_idx.keys()):
	d = os.path.join(dir, target)
	if not os.path.isdir(d):
	    continue
        images.append([])

	for root, _, fnames in sorted(os.walk(d)):
	    for fname in sorted(fnames):
		if has_file_allowed_extension(fname, extensions):
		    path = os.path.join(root, fname)
                    images[-1].append(path)

    return images


class RelightingImageFolder(data.Dataset):
    def __init__(self, root, transform=None, loader=None):
        classes, class_to_idx = self._find_classes(root)
        images = make_dataset(root, class_to_idx, IMG_EXTENSIONS)
        self.height = 256
        self.width = 256

        image_pairs = []
        for im_num, ims in enumerate(images):
            for i in range(len(ims)):
                for j in range(i+1, len(ims)):
                    image_pairs.append((ims[i], ims[j], classes[im_num],
                        os.path.splitext(os.path.basename(ims[i]))[0],
                        os.path.splitext(os.path.basename(ims[j]))[0]))

        self.image_pairs = image_pairs

        #print len(image_pairs)


    def _find_classes(self, dir):
        """
        Finds the class folders in a dataset.

        Args:
            dir (string): Root directory path.

        Returns:
            tuple: (classes, class_to_idx) where classes are relative to (dir), and class_to_idx is a dictionary.

        Ensures:
            No class is a subdirectory of another.
        """
        if sys.version_info >= (3, 5):
            # Faster and available in Python 3.5 and above
            classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        else:
            classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx

    def __len__(self):
        return len(self.image_pairs)

    def __getitem__(self, index):
        im1 = np.array(pil_loader(self.image_pairs[index][0])).astype(float)/255.0
        im2 = np.array(pil_loader(self.image_pairs[index][1])).astype(float)/255.0

        ratio = float(im1.shape[0])/float(im1.shape[1])
        if ratio > 1.73:
            h, w = 512, 256
        elif ratio < 1.0/1.73:
            h, w = 256, 512
        elif ratio > 1.41:
            h, w = 768, 512
        elif ratio < 1./1.41:
            h, w = 512, 768
        elif ratio > 1.15:
            h, w = 512, 384
        elif ratio < 1./1.15:
            h, w = 384, 512
        else:
            h, w = 512, 512
        im1 = resize(im1, (h,w))
        im2 = resize(im2, (h,w))

        return torch.from_numpy(np.transpose(im1, (2,0,1))).contiguous().float(), torch.from_numpy(np.transpose(im2,(2,0,1))).contiguous().float(), self.image_pairs[index][2], self.image_pairs[index][3], self.image_pairs[index][4]

if __name__ == '__main__':
    RIF = RelightingImageFolder('BoyadzhievImageCompositingInputs/')
    print len(RIF)
    print RIF[0][0].shape
