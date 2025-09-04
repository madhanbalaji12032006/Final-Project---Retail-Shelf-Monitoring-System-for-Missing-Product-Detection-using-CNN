import torch
from torchvision.transforms import functional as F

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

class ToTensor(object):
    def __call__(self, image, target):
        return F.to_tensor(image), target

class RandomHorizontalFlip(object):
    def __init__(self, flip_prob=0.5):
        self.flip_prob = flip_prob

    def __call__(self, image, target):
        if torch.rand(1) < self.flip_prob:
            h, w = image.shape[-2:]
            image = image.flip(-1)
            bbox = target["boxes"]
            xmin = w - bbox[:, 2]
            xmax = w - bbox[:, 0]
            bbox[:, 0] = xmin
            bbox[:, 2] = xmax
            target["boxes"] = bbox
        return image, target
