import torch
import numpy as np
import os

from torch.utils.data import DataLoader
from torch.optim import Optimizer, SGD
from torch.optim.lr_scheduler import LambdaLR
from torchvision.transforms import Normalize
from typing import List, Callable, Optional

from easy_icd.utils.losses import SimCLRLoss
from easy_icd.utils.augmentation import RandomImageAugmenter, augment_minibatch
from easy_icd.utils.early_stopping import EarlyStopper
from easy_icd.utils.datasets import compute_dataset_stats

def compute_test_loss(model: torch.nn.Module, test_dataloader: DataLoader,
		augmenter: RandomImageAugmenter, num_augments: int, normalizer: Normalize,
		loss_fn: torch.nn.Module, device: torch.device) -> torch.Tensor:
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
		device: device that the computation iwll be run on.
	"""
	model.eval()

	test_losses = []

	for j in range(10):
		images, labels = next(iter(test_dataloader))
		original_batch_size = labels.shape[0]

		augmented_minibatch = augment_minibatch(images, augmenter, num_augments, device)
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
		save_dir: str, device: torch.device) -> torch.Tensor:
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
		device: device that the training will be run on.
	"""
	train_losses = []
	test_losses = []

	model.train()

	for idx, (images, labels) in enumerate(dataloader):
		images = images.to(device)
		labels = labels.to(device)

		augmented_minibatch = augment_minibatch(images, augmenter,
			num_augments, device)

		augmented_minibatch = normalizer(augmented_minibatch)
		original_batch_size = labels.shape[0]

		features = model(augmented_minibatch)
		features_views = torch.split(features,
			[original_batch_size for i in range(num_augments)], 0)

		reshaped_features = torch.stack(features_views, 1)

		loss = loss_fn(reshaped_features, labels)
		train_losses.append(loss.item())

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		if not idx % 10:
			test_loss = compute_test_loss(model, test_dataloader, augmenter,
				num_augments, normalizer, loss_fn, device)

			test_losses.append(test_loss)

			print('Batch: {:6d} | Loss: {:.5f} | Test loss: {:.5f}'.format(idx + 1, 
				loss.item(), test_loss))

			early_stop = early_stopper.check_for_early_stop(test_loss)

			if early_stop:
				return True, train_losses, test_losses

	return False, np.mean(train_losses), np.mean(test_losses)


def get_learning_rate_scheduler(max_epochs: int, num_warmup_epochs: int,
								max_lr: float, min_lr: float):
	""" 
	Create a learning rate scheduler that first linearly increases the learning rate
	for the first max_epochs // 10 epochs, and then reduces it via cosine annealing.

	Args:
		max_epochs: int indicating the maximum number of epochs the model
			will be trained for.
		num_warmup_epochs: int indicating the number of warmup epochs.
		max_lr: float indicating the maximum learning rate.
		min_lr: float indicating the minimum learning rate.
	"""
	min_factor = min_lr / max_lr

	def lr_scheduler(epoch: int):
		"""
		Learning rate scheduler.

		Args:
			epoch: int indicating the current epoch number.
		"""
		if epoch <= num_warmup_epochs:
			return min_factor + (1 - min_factor) * (epoch) / num_warmup_epochs
		else:
			return min_factor + (1 - min_factor) * np.cos(0.5 * np.pi * (
				epoch - num_warmup_epochs) / (max_epochs - num_warmup_epochs))
		
	return lr_scheduler


def train_outlier_detector(model: torch.nn.Module, dataloader: DataLoader,
		test_dataloader: DataLoader, save_dir: str, augmenter: RandomImageAugmenter,
		num_epochs: Optional[int] = 10, optimizer: Optional[Optimizer] = None,
		num_augments: Optional[int] = 1, loss_temp: Optional[float] = 0.07,
		compute_dataset_means_and_stds: Optional[bool] = True,
		lr: Optional[float] = 0.1, min_lr: Optional[float] = 1e-3,
		num_warmup_epochs: Optional[int] = 0, losses_name: Optional[str] = '',
		gpu: Optional[bool] = False) -> List:
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
		compute_dataset_means_and_std: bool indicating whether to compute
			the per-channel means and stds over the dataset or not. Defaults to True.
		lr: float representing the maximum learning rate to be used. Defaults to 0.1.
		min_lr: float representing the minium learning rate to be used. Defaults to
			1e-3.
		num_warmup_epochs: int representing the number of epochs to warm up the 
			learning rate from min_lr to lr.
		losses_name: str indicating the name of the files in which to save
			test and train loss information.
		gpu: bool indicating whether to run on GPU or CPU. Defaults to True if
			a GPU is acessible.
	"""
	if optimizer is None:
		optimizer = SGD(model.parameters(), lr=lr, momentum=0.9)

	lr_scheduler = get_learning_rate_scheduler(num_epochs, num_warmup_epochs,
		lr, min_lr)

	scheduler = LambdaLR(optimizer, lr_scheduler, verbose=True)

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

	train_losses = []
	test_losses = []

	loss_fn = SimCLRLoss(loss_temp, True)

	device = torch.device('cuda' if gpu else 'cpu')
	loss_fn = loss_fn.to(device)
	model = model.to(device)

	for epoch_num in range(num_epochs):
		print(f'Epoch {epoch_num + 1}')

		early_stopped, train_loss, test_loss = train_epoch(model, dataloader,
			test_dataloader, optimizer, augmenter, num_augments, normalizer,
			loss_fn, early_stopper, save_dir, device)

		torch.save(model.state_dict(), os.path.join(save_dir,
			f'model_state_epoch_{epoch_num + 1}.pt'))

		train_losses.append(train_loss)
		test_losses.append(test_loss)

		scheduler.step()

		if early_stopped:
			print('Early stopping training! Test loss stopped decreasing.')
			break

	np.savetxt(os.path.join(save_dir, f'train_losses_{losses_name}.txt'), train_losses)
	np.savetxt(os.path.join(save_dir, f'test_losses_{losses_name}.txt'), test_losses)
