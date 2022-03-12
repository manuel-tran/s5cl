import cv2
import torch
import random
import numbers
import numpy as np

from PIL import Image, ImageFilter
from skimage import color
from torchvision import transforms
from torchvision.transforms import functional as F
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter

#----------------------------------------------------------------------------

# HEDJitter, AutoRandomRotation, RandomGaussBlur, RandomAffineCV2, RandomElastic
# from https://github.com/gatsby2016/Augmentation-PyTorch-Transforms

#----------------------------------------------------------------------------

class HEDJitter(object):
    """
    Randomly perturbe the HED color space value an RGB image.
    """
    def __init__(self, theta=0.): # HED_light: theta=0.05; HED_strong: theta=0.2
        assert isinstance(theta, numbers.Number), "theta should be a single number."
        self.theta = theta
        self.alpha = np.random.uniform(1-theta, 1+theta, (1, 3))
        self.betti = np.random.uniform(-theta, theta, (1, 3))

    @staticmethod
    def adjust_HED(img, alpha, betti):
        img = np.array(img)

        s = np.reshape(color.rgb2hed(img), (-1, 3))
        ns = alpha * s + betti  # perturbations on HED color space
        nimg = color.hed2rgb(np.reshape(ns, img.shape))

        imin = nimg.min()
        imax = nimg.max()
        rsimg = (255 * (nimg - imin) / (imax - imin)).astype('uint8')  # rescale to [0,255]
        # transfer to PIL image
        return Image.fromarray(rsimg)

    def __call__(self, img):
        return self.adjust_HED(img, self.alpha, self.betti)

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += 'theta={0}'.format(self.theta)
        format_string += ',alpha={0}'.format(self.alpha)
        format_string += ',betti={0}'.format(self.betti)
        return format_string

#----------------------------------------------------------------------------

class AutoRandomRotation(object):
    """
    Randomly select angle 0, 90, 180 or 270 for rotating the image.
    """

    def __init__(self, degree=None, resample=False, expand=True, center=None, fill=0):
        if degree is None:
            self.degrees = random.choice([0, 90, 180, 270])
        else:
            assert degree in [0, 90, 180, 270], 'degree must be in [0, 90, 180, 270]'
            self.degrees = degree

        self.resample = resample
        self.expand = expand
        self.center = center
        self.fill = fill

    def __call__(self, img):
        return F.rotate(img, self.degrees, self.resample, self.expand, self.center, self.fill)

    def __repr__(self):
        format_string = self.__class__.__name__ + '(degrees={0}'.format(self.degrees)
        format_string += ', resample={0}'.format(self.resample)
        format_string += ', expand={0}'.format(self.expand)
        if self.center is not None:
            format_string += ', center={0}'.format(self.center)
        format_string += ')'
        return format_string

#----------------------------------------------------------------------------

class RandomGaussBlur(object):
    """
    Randomly apply Gaussian blurring on image by radius parameter.
    """
    def __init__(self, radius=None):
        if radius is not None:
            assert isinstance(radius, (tuple, list)) and len(radius) == 2, \
                "radius should be a list or tuple and it must be of length 2."
            self.radius = random.uniform(radius[0], radius[1])
        else:
            self.radius = 0.0

    def __call__(self, img):
        return img.filter(ImageFilter.GaussianBlur(radius=self.radius))

    def __repr__(self):
        return self.__class__.__name__ + '(Gaussian Blur radius={0})'.format(self.radius)

#----------------------------------------------------------------------------

class RandomAffineCV2(object):
    """
    Randomly apply affine transformation by CV2 method on image by alpha parameter.
    """
    def __init__(self, alpha):
        assert isinstance(alpha, numbers.Number), "alpha should be a single number."
        assert 0. <= alpha <= 0.15, \
            "In pathological image, alpha should be in (0,0.15), you can change in myTransform.py"
        self.alpha = alpha

    @staticmethod
    def affineTransformCV2(img, alpha, mask=None):
        alpha = img.shape[1] * alpha
        if mask is not None:
            mask = np.array(mask).astype(np.uint8)
            img = np.concatenate((img, mask[..., None]), axis=2)

        imgsize = img.shape[:2]
        center = np.float32(imgsize) // 2
        censize = min(imgsize) // 3
        pts1 = np.float32([center+censize, [center[0]+censize, center[1]-censize], center-censize])  # raw point
        pts2 = pts1 + np.random.uniform(-alpha, alpha, size=pts1.shape).astype(np.float32)  # output point
        M = cv2.getAffineTransform(pts1, pts2)  # affine matrix
        img = cv2.warpAffine(img, M, imgsize[::-1],
                               flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT_101)
        if mask is not None:
            return Image.fromarray(img[..., :3]), Image.fromarray(img[..., 3])
        else:
            return Image.fromarray(img)

    def __call__(self, img, mask=None):
        return self.affineTransformCV2(np.array(img), self.alpha, mask)

    def __repr__(self):
        return self.__class__.__name__ + '(alpha value={0})'.format(self.alpha)

#----------------------------------------------------------------------------

class RandomElastic(object):
    """
    Randomly apply elastic transformation by CV2 method on image by alpha, sigma parameter.
    """
    def __init__(self, alpha, sigma):
        assert isinstance(alpha, numbers.Number) and isinstance(sigma, numbers.Number), \
            "alpha and sigma should be a single number."
        assert 0.05 <= sigma <= 0.1, \
            "In pathological image, sigma should be in (0.05,0.1)"
        self.alpha = alpha
        self.sigma = sigma

    @staticmethod
    def RandomElasticCV2(img, alpha, sigma, mask=None):
        alpha = img.shape[1] * alpha
        sigma = img.shape[1] * sigma
        if mask is not None:
            mask = np.array(mask).astype(np.uint8)
            img = np.concatenate((img, mask[..., None]), axis=2)

        shape = img.shape

        dx = gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma) * alpha
        dy = gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma) * alpha
        # dz = np.zeros_like(dx)

        x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
        indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1)), np.reshape(z, (-1, 1))

        img = map_coordinates(img, indices, order=0, mode='reflect').reshape(shape)
        if mask is not None:
            return Image.fromarray(img[..., :3]), Image.fromarray(img[..., 3])
        else:
            return Image.fromarray(img)

    def __call__(self, img, mask=None):
        return self.RandomElasticCV2(np.array(img), self.alpha, self.sigma, mask)

    def __repr__(self):
        format_string = self.__class__.__name__ + '(alpha value={0})'.format(self.alpha)
        format_string += ', sigma={0}'.format(self.sigma)
        format_string += ')'
        return format_string
    
#----------------------------------------------------------------------------

class RandomRotate90:
    def __init__(self, angles):
        self.angles = angles

    def __call__(self, x):
        angle = random.choice(self.angles)
        return transforms.functional.rotate(x, angle)

#----------------------------------------------------------------------------

class RandomResizeRange:
    def __init__(self, lower, upper):
        self.lower = lower
        self.upper = upper

    def __call__(self, x):
        size = random.randint(self.lower, self.upper)
        return transforms.functional.resize(x, size)

#----------------------------------------------------------------------------

class Transform: 
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std 
        
        self.transform_weak = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(), 
                transforms.RandomVerticalFlip(), 
                RandomRotate90([0, 90, 180, 270]),
                transforms.ColorJitter(0.1, 0.1, 0.1, 0.1), 
                transforms.ToTensor(), 
                transforms.Normalize(self.mean, self.std)
            ]
        )
        
        self.transform_strong = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(0.5),
                transforms.RandomVerticalFlip(0.5),
                RandomRotate90([0, 90, 180, 270]),
                HEDJitter(theta=0.01),
                transforms.RandomResizedCrop(224),
                RandomAffineCV2(alpha=0.05),
                transforms.ToTensor(), 
                transforms.Normalize(self.mean, self.std),
            ]
        )

    def __call__(self, x):
        y1 = self.transform_weak(x)
        y2 = self.transform_strong(x)
        return y1, y2