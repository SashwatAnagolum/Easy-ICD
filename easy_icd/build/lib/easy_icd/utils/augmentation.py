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
				 transform_probs: Optional[Union[torch.Tensor, List,
				 np.ndarray]] = None, min_transforms: Optional[int] = 1) -> None:
		"""
		Constructor for RandomImageAugmenter objects.

		Args:
			output_image_size: tuple (width, height) indicating output image size.
			transform_probs: Array-like of probabilities for each transformation
				that could be applied.
			min_transforms: int representing the minimum number of
				transformations to apply, default 0.
			scale_images: bool indicating whether to scale images down to [0, 1]
				from [0, 255] or not. Default is True.
		"""
		if transform_probs is None:
			transform_probs = [0.1 for i in range(8)]
		
		self.transform_probs = transform_probs
		self.output_image_size = output_image_size
		self.min_transforms = min_transforms
		
		self.transforms = [RandomHorizontalFlip(p=1),
			RandomVerticalFlip(p=1),
			RandomGrayscale(p=1),
			RandomErasing(p=1, scale=(0.01, 0.15)),
			ColorJitter(0.1, 0.1, 0.1, 0.1),
			GaussianBlur(3, (0.05, 0.5)),
			RandomResizedCrop(output_image_size, (0.8, 1)),
			RandomRotation(15)]

		self.num_transforms = len(self.transforms)       	
		
	def augment(self, images: torch.Tensor) -> torch.Tensor:
		"""
		Randomly apply some transformations to the minibatch of images passed in.

		Args:
			images: Tensor of dimensions [batch_size, 3, width, height] containing
				the images to be augmented.

		Returns:
			An augmented copy of the images.
		"""
		applied_transforms = []
		images_copy = images.clone().detach()
		
		ordering = torch.randperm(self.num_transforms)
		
		for i in range(self.num_transforms):
			if torch.rand(1) < self.transform_probs[ordering[i]]:
				images_copy = self.transforms[ordering[i]](images_copy)
				applied_transforms.append(ordering[i])

		if len(applied_transforms) < self.min_transforms:
			unapplied_transforms = [i for i in range(
				self.num_transforms) if i not in applied_transforms]

			num_extra_transforms = self.min_transforms - len(applied_transforms)
			extra_transforms = np.random.choice(unapplied_transforms,
				num_extra_transforms, False)

			for i in range(num_extra_transforms):
				images_copy = self.transforms[extra_transforms[i]](images_copy)
				applied_transforms.append(extra_transforms[i])
					
		return images_copy


def augment_minibatch(minibatch: torch.Tensor, augmenter: RandomImageAugmenter,
					  num_augments: int) -> torch.Tensor:
	"""
	Augment a minibatch of images and return a multi-viewed batch of images.

	Args:
		minibatch: torch.Tensor representing a minibatch of images.
		augmenter: RandomImageAugmenter to be used to augment the images.
		num_augments: int representing the number of times to augment the images.
			The returned minibatch size will be (1 + num_augments) * original_bsz,
			where original_bsz is the original batch size of the minibatch.

	Returns:
		A multi-viewed batch of images.
	"""
	augmented_minibatch = []

	for i in range(num_augments):
		augmented_minibatch.append(augmenter.augment(minibatch))

	return torch.cat(augmented_minibatch, 0)