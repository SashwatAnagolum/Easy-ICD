import torch
import numpy as np
import os

from torch.utils.data import DataLoader
from torch.optim import Optimizer, SGD
from torchvision.transforms import Normalize
from typing import List, Callable, Optional

from easy_icd.utils.losses import SimCLRLoss
from easy_icd.utils.augmentation import RandomImageAugmenter, augment_minibatch
from easy_icd.utils.early_stopping import EarlyStopper
from easy_icd.utils.datasets import compute_dataset_stats

def compute_test_loss(model: torch.nn.Module, test_dataloader: DataLoader,
		augmenter: RandomImageAugmenter, num_augments: int, normalizer: Normalize,
		loss_fn: torch.nn.Module) -> torch.Tensor:
	"""
	Compute the test loss for the model over a few minibatches from the test dataset.

	Args:
		model: torch.nn.Module representing the outlier detector to be trained.
		test_dataloader: DataLoader that fetches images from a held-out test portion
			of the dataset to be cleaned.
		augmenter: the RandomImageAugmenter to be used to augment image minibatches.
		num_augments: int > 2 representing the number of views to create for each image
			in a minibatch.
		normalizer: Normalize transform used to normalize the images.
		loss_fn: torch.nn.Module representing the SimCLR loss to be used during
			training.
	"""
	model.eval()

	test_losses = []

	for j in range(10):
		images, labels = next(iter(test_dataloader))
		original_batch_size = labels.shape[0]

		augmented_minibatch = augment_minibatch(images, augmenter, num_augments)
		augmented_minibatch = normalizer(augmented_minibatch)

		features = model(augmented_minibatch)
		features_views = torch.split(features,
				[original_batch_size for i in range(num_augments)], 0)

		reshaped_features = torch.stack(features_views, 1)

		loss = loss_fn(reshaped_features, labels)	

		test_losses.append(loss.detach().item())

	mean_test_loss = np.mean(test_losses)	

	model.train()

	return mean_test_loss


def train_epoch(model: torch.nn.Module, dataloader: DataLoader,
		test_dataloader: DataLoader, optimizer: Optimizer,
		augmenter: RandomImageAugmenter, num_augments: int, normalizer: Normalize,
		loss_fn: torch.nn.Module, early_stopper: EarlyStopper,
		save_dir: str) -> torch.Tensor:
	"""
	Train the outlier detector for one epoch.

	Args:
		model: torch.nn.Module representing the outlier detector to be trained.
		dataloader: DataLoader that fetches images from the dataset to be cleaned.
		test_dataloader: DataLoader that fetches images from a held-out test portion
			of the dataset to be cleaned.
		optimizer: Optimizer to be used during training.
		augmenter: the RandomImageAugmenter to be used to augment image minibatches.
		num_augments: int > 2 representing the number of views to create for each image
			in a minibatch.
		normalizer: Normalize transform used to normalize the images.
		loss_fn: torch.nn.Module representing the SimCLR loss to be used during
			training.
		early_stopper: EarlyStopper used to check whether we need to stop training
			to prevent overfitting or not.
		save_dir: str path to the folder the model needs to be saved in.
	"""
	model.train()

	for idx, (images, labels) in enumerate(dataloader):
		augmented_minibatch = augment_minibatch(images, augmenter, num_augments)
		augmented_minibatch = normalizer(augmented_minibatch)
		original_batch_size = labels.shape[0]

		features = model(augmented_minibatch)
		features_views = torch.split(features,
			[original_batch_size for i in range(num_augments)], 0)

		reshaped_features = torch.stack(features_views, 1)

		loss = loss_fn(reshaped_features, labels)

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		if not idx % 10:
			test_loss = compute_test_loss(model, test_dataloader, augmenter,
				num_augments, normalizer, loss_fn)

			print('Batch: {:6d} | Loss: {:.5f} | Test loss: {:.5f}'.format(idx + 1, 
				loss.item(), test_loss))

			early_stop = early_stopper.check_for_early_stop(test_loss)

			if early_stop:
				torch.save(model.state_dict(), os.path.join(save_dir, 'model_state.pt'))

				return early_stop

	return False


def train_outlier_detector(model: torch.nn.Module, dataloader: DataLoader,
		test_dataloader: DataLoader, save_dir: str, augmenter: RandomImageAugmenter,
		num_epochs: Optional[int] = 10, optimizer: Optional[Optimizer] = None,
		num_augments: Optional[int] = 1, loss_temp: Optional[float] = 0.07,
		compute_dataset_means_and_stds: Optional[bool] = True) -> List:
	"""
	Train an outlier detector.

	Args:
		model: torch.nn.Module representing the outlier detector to be trained.
		dataloader: DataLoader that fetches images from the dataset to be cleaned.
		test_dataloader: DataLoader that fetches images from a held-out test portion
			of the dataset to be cleaned.
		augmenter: the RandomImageAugmenter to be used to augment image mnibatches.
		num_epochs: int indicating the number of epochs of training. Defaults to 10.
		optimizer: Optimizer to be used during training. Defaults to SGD with lr 0.01,
			momentum 0.9.
		num_augments: int representing the number of views to create for each image
			in a minibatch. Defaults to 1. 
		loss_temp: float representing the temperature to be used with the loss function
			for training. Defaults to 0.07.   
	"""
	if optimizer is None:
		optimizer = SGD(model.parameters(), lr=1e-1, momentum=0.9)

	if compute_dataset_means_and_stds:
		means, stds = compute_dataset_stats(dataloader, 50)
		print(f'Estimated statistics for channels:\nMeans: {means}\nStd dev.: {stds}')
	else:
		means = [0.5, 0.5, 0.5]
		stds = [0.5, 0.5, 0.5]

	normalizer = Normalize(means, stds)
	early_stopper = EarlyStopper(10)

	if not os.path.exists(save_dir):
		os.makedirs(save_dir)

	loss_fn = SimCLRLoss(loss_temp, True)

	for epoch_num in range(num_epochs):
		early_stopped = train_epoch(model, dataloader, test_dataloader, optimizer,
			augmenter, num_augments, normalizer, loss_fn, early_stopper, save_dir)

		if early_stopped:
			print('Early stopping training! Test loss stopped decreasing.')
			break
		else:
			torch.save(model.state_dict(), os.path.join(save_dir, 'model_state.pt'))
