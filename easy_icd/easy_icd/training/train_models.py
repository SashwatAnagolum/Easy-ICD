"""
Author: Sashwat Anagolum
"""

import torch
import numpy as np
import os

from torch.utils.data import DataLoader
from torch.optim import Optimizer, SGD
from torch.optim.lr_scheduler import LambdaLR
from torchvision.transforms import Normalize
from typing import List, Callable, Optional

from easy_icd.training.losses import SimCLRLoss, CELoss
from easy_icd.utils.augmentation import RandomImageAugmenter, augment_minibatch
from easy_icd.training.early_stopping import EarlyStopper
from easy_icd.utils.datasets import compute_dataset_stats

def compute_accuracy(predictions: torch.Tensor, labels: torch.Tensor,
					 loss_type: str):
	"""
	Compute the accuracy of a model.

	Args:
		predictions (torch.Tensor): model predictions.
		labels (torch.Tensor): ground truth.
		loss_type (str): whether to compute constrastive accuracy
			or classification accuracy. Currently only supports classification
			accuracy, so all calls with loss_type == 'simclr' return 0.

	Returns:
		(float): minibatch accuracy.
	"""
	if loss_type == 'ce':
		original_batch_size = labels.shape[0]
		num_samples = predictions.shape[0]
		labels = labels.repeat(num_samples // original_batch_size)

		model_preds = torch.argmax(predictions.detach(), 1)
		accuracy = torch.sum(torch.eq(model_preds, labels)) / num_samples

		return accuracy.item()
	else:
		return 0


def compute_test_loss_and_accuracy(model: torch.nn.Module, test_dataloader: DataLoader,
								   augmenter: RandomImageAugmenter, num_augments: int,
								   normalizer: Normalize, loss_fn: torch.nn.Module,
								   loss_type: str, device: torch.device) -> torch.Tensor:
	"""
	Compute the test loss and accuray for the model over a few minibatches from
	the test dataset.

	Args:
		model (nn.Module): the outlier detector to be trained.
		test_dataloader (DataLoader): DataLoader that fetches images from a held-out
			test portion of the dataset to be cleaned.
		augmenter (RandomImageAugmenter): the RandomImageAugmenter to be used to
			augment image minibatches.
		num_augments (int): the number of views to create for each image
			in a minibatch.
		normalizer (Normalize): transform used to normalize the images.
		loss_fn (nn.Module): the loss to be used during
			training.
		loss_type (str): which type of loss to compute.
		device (torch.device): device that the computation will be run on.

	Returns:
		(float): test set loss
		(float): test set accuracy
	"""
	model.eval()

	test_losses = []
	test_accs = []

	for j in range(10):
		images, labels = next(iter(test_dataloader))
		images = images.to(device)
		labels = labels.to(device)

		original_batch_size = labels.shape[0]

		if loss_type == 'simclr':
			augmented_minibatch = augment_minibatch(images, augmenter,
				num_augments, device)
		elif loss_type == 'ce':
			augmented_minibatch = images

		augmented_minibatch = normalizer(augmented_minibatch)

		features = model(augmented_minibatch)

		loss = loss_fn(features, labels)   

		test_losses.append(loss.detach().item())
		test_accs.append(compute_accuracy(features, labels, loss_type))

	mean_test_loss = np.mean(test_losses)  
	mean_test_acc = np.mean(test_accs) 

	model.train()

	return mean_test_loss, mean_test_acc


def train_epoch(model: torch.nn.Module, dataloader: DataLoader,
				test_dataloader: DataLoader, optimizer: Optimizer,
				augmenter: RandomImageAugmenter, num_augments: int,
				normalizer: Normalize, loss_fn: torch.nn.Module, 
				early_stopper: EarlyStopper, save_dir: str, device: torch.device,
				loss_type: Optional[str] = 'simclr') -> torch.Tensor:
	"""
	Train the outlier detector for one epoch.

	Args:
		model (nn.Module): the model to be trained.
		dataloader (DataLoder): DataLoader that fetches images from the dataset to be
			cleaned.
		test_dataloader (DataLoader): DataLoader that fetches images from a held-out
			test portion of the dataset to be cleaned.
		optimizer (Optimizer): optimizer to be used during training.
		augmenter (RandomImageAugmenter): the RandomImageAugmenter to be used to
			augment image minibatches.
		num_augments (int): the number of views to create for each image
			in a minibatch.
		normalizer (Normalize): transform used to normalize the images.
		loss_fn (nn.Module): the loss to be used during training.
		early_stopper (EarlyStopper): used to check whether we need to stop training
			to prevent overfitting or not.
		save_dir (str): path to the folder the model needs to be saved in.
		device: device that the training will be run on.
		loss_type (str, optional): which type of loss to compute. 'simclr' results in the
			SimCLR being used. 'ce' will train the model using the cross-entropy loss.
	"""
	train_losses = []
	train_accs = []
	test_losses = []
	test_accs = []

	model.train()

	for idx, (images, labels) in enumerate(dataloader):
		images = images.to(device)
		labels = labels.to(device)

		augmented_minibatch = augment_minibatch(images, augmenter,
			num_augments, device)

		augmented_minibatch = normalizer(augmented_minibatch)
		original_batch_size = labels.shape[0]

		features = model(augmented_minibatch)

		loss = loss_fn(features, labels)
		acc = compute_accuracy(features, labels, loss_type)

		train_losses.append(loss.item())
		train_accs.append(acc)

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		if not idx % 10:
			test_loss, test_acc = compute_test_loss_and_accuracy(model,
				test_dataloader, augmenter, num_augments, normalizer, loss_fn,
				loss_type, device)

			test_losses.append(test_loss)
			test_accs.append(test_acc)

			print('Batch: {:4d} | Loss: {:9.5f} | Test loss: {:9.5f}'.format(idx + 1,
				loss.item(), test_loss) + ' | Acc: {:9.5f} | Test acc: {:9.5f}'.format(
				acc, test_acc))

			early_stop = early_stopper.check_for_early_stop(test_loss)

			if early_stop:
				return (True, np.mean(train_losses), np.mean(test_losses),
					np.mean(train_accs), np.mean(test_accs))

	return (False, np.mean(train_losses), np.mean(test_losses), np.mean(train_accs),
		np.mean(test_accs))


def get_learning_rate_scheduler(max_epochs: int, num_warmup_epochs: int,
								max_lr: float, min_lr: float):
	""" 
	Create a learning rate scheduler that first linearly increases the learning rate
	for the first few epochs, and then reduces it via cosine annealing.

	Args:
		max_epochs (int): the maximum number of epochs the model
			will be trained for.
		num_warmup_epochs (int): the number of warmup epochs.
		max_lr (float): the maximum learning rate.
		min_lr (float): the minimum learning rate.
	"""
	min_factor = min_lr / max_lr

	def lr_scheduler(epoch: int):
		"""
		Learning rate scheduler.

		Args:
			epoch (int): the current epoch number.
		"""
		if epoch <= num_warmup_epochs:
			return min_factor + (1 - min_factor) * (epoch) / num_warmup_epochs
		else:
			return min_factor + (1 - min_factor) * np.cos(0.5 * np.pi * (
				epoch - num_warmup_epochs) / (max_epochs - num_warmup_epochs))
		
	return lr_scheduler


def train_model(model: torch.nn.Module, dataloader: DataLoader,
		test_dataloader: DataLoader, save_dir: str, augmenter: RandomImageAugmenter,
		loss_type: str, num_epochs: Optional[int] = 10,
		optimizer: Optional[Optimizer] = None,
		num_augments: Optional[int] = 1, loss_temp: Optional[float] = 0.07,
		compute_dataset_means_and_stds: Optional[bool] = True,
		lr: Optional[float] = 0.1, min_lr: Optional[float] = 1e-3,
		num_warmup_epochs: Optional[int] = 0, losses_name: Optional[str] = '',
		gpu: Optional[bool] = False, epoch_offset: Optional[int] = 0,
		dataset_means_and_stds: Optional[List[List[float]]] = None,
		simclr_use_labels: Optional[bool] = True,
		num_stat_counts: Optional[int] = 50) -> List:
	"""
	Train an outlier detector.

	Args:
		model (nn.Module): the model to be trained.
		dataloader (DataLoder): DataLoader that fetches images from the dataset to be
			cleaned.
		test_dataloader (DataLoader): DataLoader that fetches images from a held-out
			test portion of the dataset to be cleaned.
		save_dir (str): path to the folder the model needs to be saved in.
		augmenter (RandomImageAugmenter): the RandomImageAugmenter to be used to
			augment image minibatches.
		loss_type (str, optional): str indicating what loss to use for training.
			'simclr' results in using the modified SimCLR loss, and 'ce' results in
			using the standard supervised log loss.
		num_epochs (int,, optional): the number of epochs of training. Defaults to 10.
		optimizer (Optimizer, optional): optimizer to be used during training.
			Defaults to SGD with lr = 0.1 and momentum = 0.9.
		num_augments (int, optional): the number of views to create for each image
			in a minibatch.
		loss_temp (float, optional): the temperature to be used with the loss function
			for training. Defaults to 0.07. Ignored if the loss_type is 'ce'. 
		compute_dataset_means_and_std (bool): whether to compute
			the per-channel means and stds over the dataset or not. Defaults to True.
		lr (float, optional): the maximum learning rate to be used. Defaults to 0.1.
		min_lr (float, optional): the minium learning rate to be used. Defaults to
			1e-3.
		num_warmup_epochs (int, optional): the number of epochs to warm up the 
			learning rate from min_lr to lr. Defaults to 0.
		losses_name (str, optional): the name of the files in which to save
			test and train loss information.
		gpu  (bool, optional): bool indicating whether to run on GPU or CPU.
			Defaults to True if a GPU is acessible.
		epoch_offset (int, optional): int representing the number to start counting
			epochs from. Defaults to 0.
		dataset_means_and_stds (list, optional): statistics of the dataset,
			overriding the compute_dataset_means_and_stds parameter. Defaults to None.
		simclr_use_labels (bool, optional): whether to use labels with the 
			SimCLR loss or not. Defaults to True.
		num_stat_counts (int, optional): how many minibatches to average
			dataset statistics over, if computed.
	"""
	if optimizer is None:
		optimizer = SGD(model.parameters(), lr=lr, momentum=0.9)

	lr_scheduler = get_learning_rate_scheduler(num_epochs, num_warmup_epochs,
		lr, min_lr)

	scheduler = LambdaLR(optimizer, lr_scheduler, verbose=True)

	if dataset_means_and_stds is None:
		if compute_dataset_means_and_stds:
			means, stds = compute_dataset_stats(dataloader, num_stat_counts)
			print('Estimated statistics for channels:')
			print(f'Means: {means}\nStd dev.: {stds}')
		else:
			means = [0.5, 0.5, 0.5]
			stds = [0.5, 0.5, 0.5]
	else:
		means = dataset_means_and_stds[0]
		stds = dataset_means_and_stds[1]

	normalizer = Normalize(means, stds)
	early_stopper = EarlyStopper(10)

	if not os.path.exists(save_dir):
		os.makedirs(save_dir)

	if loss_type == 'simclr':
		loss_fn = SimCLRLoss(loss_temp, True, True, simclr_use_labels)
	elif loss_type == 'ce':
		loss_fn = CELoss()

	device = torch.device('cuda' if gpu else 'cpu')
	loss_fn = loss_fn.to(device)
	model = model.to(device)

	train_losses_file = open(os.path.join(save_dir, f'train_losses_{losses_name}.txt'),
		'a')

	test_losses_file = open(os.path.join(save_dir, f'test_losses_{losses_name}.txt'),
		'a')

	for epoch_num in range(epoch_offset, num_epochs + epoch_offset):
		print(f'Epoch {epoch_num + 1}')

		early_stopped, train_loss, test_loss, train_acc, test_acc = train_epoch(model,
			dataloader, test_dataloader, optimizer, augmenter, num_augments, normalizer,
			loss_fn, early_stopper, save_dir, device, loss_type)

		torch.save(model.state_dict(), os.path.join(save_dir,
			f'model_state_epoch_{epoch_num + 1}.pt'))

		print('Train Loss: {:.5f} | Test Loss: {:.5f} | Train Acc: {:.5f}'.format(
			train_loss, test_loss, train_acc) + '  | Test Acc: {:.5f}'.format(test_acc))

		train_losses_file.write(f'Epoch {epoch_num + 1}: {train_loss} | {train_acc}\n')
		test_losses_file.write(f'Epoch {epoch_num + 1}: {test_loss} | {test_acc}\n')
		train_losses_file.flush()
		test_losses_file.flush()

		scheduler.step()

		if early_stopped:
			print('Early stopping training! Test loss stopped decreasing.')
			break

	train_losses_file.close()
	test_losses_file.close()
