import torch
import numpy as np

from torchvision.transforms import (RandomHorizontalFlip, RandomVerticalFlip,
    RandomGrayscale, ColorJitter, GaussianBlur, RandomRotation, RandomResizedCrop,
    RandomPosterize, RandomErasing)

from typing import Optional, Tuple, List, Union

class RandomImageAugmenter():
    """
    Randomly apply image augmentations.
    """
    def __init__(self, output_image_size: Tuple[int, int],
                 transform_probs: Optional[Unionp[torch.Tensor, List,
                 np.ndarray]] = None):
        """
        Constructor for RandomImageAugmenter objects.

        Args:
            output_image_size: tuple (width, height) indicating output image size.
            transform_probs: Array-like of probabilities for each transformation
                that could be applied.
        """
        if transform_probs is None:
            transform_probs = [0.2 for i in range(9)]
        
        self.transform_probs = transform_probs
        self.output_image_size = output_image_size
        
        self.probabilistic_transforms = [RandomHorizontalFlip(p=self.transform_probs[0]),
            RandomVerticalFlip(p=self.transform_probs[1]),
            RandomGrayscale(p=self.transform_probs[2]),
            RandomErasing(p=self.transform_probs[3], scale=(0.01, 0.15)),
            RandomPosterize(p=self.transform_probs[4], bits=6)]
    
        self.deterministic_transforms = [ColorJitter(0.25, 0.25, 0.25, 0.15),
            GaussianBlur(5, (0.1, 1)), RandomResizedCrop(output_image_size, (0.6, 1)),
            RandomRotation(30)]
        
        self.transforms = self.probabilistic_transforms
        self.transforms += self.deterministic_transforms
        self.num_probabilistic_transforms = len(self.probabilistic_transforms)
        self.num_transforms = len(self.transforms)
        
    def augment(self, images: torch.Tensor):
        """
        Randomly apply some transformations to the minibatch of images passed in.

        Args:
            images: Tensor of dimensions [batch_size, 3, width, height] containing
                the images to be augmented.

        Returns:
            An augmented copy of the images.
        """
        images_copy = images.clone().detach()
        
        ordering = torch.randperm(self.num_transforms)
        
        for i in range(self.num_transforms):
            if ordering[i] < self.num_probabilistic_transforms:
                images_copy = self.transforms[ordering[i]](images_copy)
            else:
                if torch.rand(1) < self.transform_probs[ordering[i]]:
                    images_copy = self.transforms[ordering[i]](images_copy)
                    
        return images_copy