"""
Author: Sashwat Anagolum
"""

import torch
import numpy as np
import torch.nn as nn
import json
import os

from torch.utils.data import DataLoader
from typing import Optional, List, Tuple
from torchvision.transforms import Normalize

from easy_icd.utils.augmentation import RandomImageAugmenter, augment_minibatch
from easy_icd.training.losses import SimCLRLoss
from easy_icd.utils.datasets import create_dataset

def compute_sample_hardness(model: nn.Module, class_dataloader: DataLoader,
							augmenter: RandomImageAugmenter, num_augments: int,
							num_loss_samples: int, loss_fn: nn.Module,
							normalizer: Normalize, device: torch.device) -> torch.Tensor:
	"""
	Compute sample 'hardness' via the expected SimCLR Loss over multiple sets of views
	of a sample.

	Args:
		model (nn.Module): the trained model to use.
		class_dataloader (DataLoader): DataLoader fetching images from the class of
			interest. Must have shuffle == False, since this function relies on the
			ordering of the images being fetched being the same as the ordering of the
			images in the folder they are stored in.
		augmenter (RandomImageAugmenter): RandomImageAugmenter used to augment images.
		num_augments (int): the number of views of each minibatch to pass
			into the network.
		num_loss_samples (int): the number of loss values to average over for
			each sample.
		loss_fn (nn.Module): the loss function to use.
		normalizer (Normalize): a transform used to normalize the augmented images.
		device (torch.device): the device the computation will happen on.
	Returns:
		(np.ndarray): sample hardness scores.
	"""
	sample_losses = []

	with torch.no_grad():
		for idx, (images, labels) in enumerate(class_dataloader):
			curr_sample_losses = []
			batch_size = labels.shape[0]

			for i in range(num_loss_samples):
				augmented_images = augment_minibatch(images, augmenter, num_augments,
					device)

				features = model(normalizer(augmented_images))
				losses = loss_fn(features, labels)[:batch_size]

				curr_sample_losses.append(losses.cpu().detach().numpy())

			mean_sample_losses = np.mean(np.array(curr_sample_losses), 0)
			sample_losses.append(mean_sample_losses)

	return np.concatenate(sample_losses)


def compute_sample_proximity_and_redundancy(model: nn.Module,
											class_dataloader: DataLoader,
									        normalizer: Normalize,
									        device: torch.device,
									        batch_size: int):
	"""
	Compute the proximity and redundancy of samples within a minibatch.

	Args:
		model (nn.Module): the trained model to use.
		class_dataloader (DataLoader): DataLoader fetching images from the class of
			interest. Must have shuffle == False, since this function relies on the
			ordering of the images being fetched being the same as the ordering of the
			images in the folder they are stored in.
		normalizer (Normalize): a transform used to normalize the augmented images.
		device (torch.device): the device the computation will happen on.
		batch_size (int): size of loaded minibatches.

	Returns:
		(np.ndarray): sample proximity scores
		(np.ndarray): sample redundancy scores
		(np.ndarray): sample redundant pair indices
	"""
	sample_redundancies = []
	sample_proximities = []
	sample_redundant_pairs = []

	with torch.no_grad():
		for idx, (images, labels) in enumerate(class_dataloader):
			curr_batch_size = labels.shape[0]

			images = normalizer(images).to(device)
			features = model(images)
			normalized_features = torch.div(features,
				torch.linalg.norm(features, dim=1, keepdim=True))

			similarities = torch.matmul(normalized_features, normalized_features.T)
			upper_similarities = torch.sub(torch.triu(similarities),
				2 * torch.eye(curr_batch_size).to(device))

			curr_sample_redundancies, curr_redundant_pairs = torch.max(
				upper_similarities, dim=1)

			curr_sample_redundancies = curr_sample_redundancies.cpu().detach().numpy()
			curr_redundant_pairs = curr_redundant_pairs.cpu().detach().numpy()
			curr_redundant_pairs += batch_size * idx

			curr_sample_proximities = torch.mean(similarities, dim=1).cpu()

			sample_redundancies.append(curr_sample_redundancies)
			sample_redundant_pairs.append(curr_redundant_pairs)
			sample_proximities.append(curr_sample_proximities.detach().numpy())

	sample_proximities = np.concatenate(sample_proximities)
	sample_redundancies = np.concatenate(sample_redundancies)
	sample_redundant_pairs = np.concatenate(sample_redundant_pairs)

	return sample_proximities, sample_redundancies, sample_redundant_pairs


def analyze_data(model: nn.Module, data_dir: str, class_names: List[str],
				 dataset_means_and_stds: list[List[float]], image_size: Tuple[int, int],
				 num_loss_samples: Optional[int] = 10, 
				 gpu: Optional[bool] = False) -> None:
	"""
	Analyze the images in every class of the scraped dataset using intra-class
	sample proximity, redundancy, and sample hardness.

	Args:
		model (nn.Module): the trained model to use.
		data_dir (str): the folder where the dataste is stored.
		class_names (list): the names of the classes to analyze.
		dataset_means_and_stds (list): the per-channel
			means and stds of the images, used for normalization.
		image_size (tuple): side lengths of the images in pixels.
		num_loss_samples (int, optional): the number of loss values to average over for
			each sample. Defaults to 10.
		gpu (bool, optional): bool indicating whether to use the GPU or not.
			Defaults to False.
	"""
	device = torch.device('cuda' if gpu else 'cpu')
	model.to(device)

	if dataset_means_and_stds is None:
		dataloader = create_dataset(data_dir, class_names, False, True)
		means, stds = compute_dataset_stats(dataloader, 50)
	else:
		means = dataset_means_and_stds[0]
		stds = dataset_means_and_stds[1]

	normalizer = Normalize(means, stds)
	augmenter = RandomImageAugmenter(image_size, 0.2 * torch.ones(8), 2)
	loss_fn = SimCLRLoss(1, True, False, False, True)

	model.eval()

	class_names = [class_name.replace(' ', '_') for class_name in class_names]

	for class_name in class_names:
		print(f'Analyzing images in class: {class_name}')

		class_dir = os.path.join(data_dir, class_name)

		class_dataset = create_dataset(data_dir, [class_name], False, True)
		class_num_samples = json.loads(open(os.path.join(
			class_dir, 'class_scraping_info.json'),
			'r').read())['num_saved_images']

		batch_size = min(class_num_samples, 512)
		class_dataloader = DataLoader(class_dataset, batch_size=batch_size,
			shuffle=False)

		hardness_scores = compute_sample_hardness(model, class_dataloader,
			augmenter, 2, num_loss_samples, loss_fn, normalizer, device)

		proximity_scores, redundancy_scores, redundant_pairs =\
			compute_sample_proximity_and_redundancy(model, class_dataloader, normalizer,
				device, batch_size)

		hardness_scores -= np.min(hardness_scores)
		hardness_scores /= np.max(hardness_scores)

		proximity_scores -= np.min(proximity_scores)
		proximity_scores /= np.max(proximity_scores)

		redundancy_scores = 1 - redundancy_scores
		redundancy_scores -= np.min(redundancy_scores)
		redundancy_scores /= np.max(redundancy_scores)

		np.save(os.path.join(class_dir, 'hardness_scores.npy'), hardness_scores)
		np.save(os.path.join(class_dir, 'proximity_scores.npy'), proximity_scores)
		np.save(os.path.join(class_dir, 'redundancy_scores.npy'), redundancy_scores)
		np.save(os.path.join(class_dir, 'redundant_pairs.npy'), redundant_pairs)
