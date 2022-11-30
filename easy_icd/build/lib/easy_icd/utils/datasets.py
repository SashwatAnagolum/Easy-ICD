import torch
import numpy as np
import os
import json

from torch.utils.data import Dataset, DataLoader, RandomSampler
from torchvision.io import read_image
from torchvision.transforms import Lambda
from typing import Optional, List, Tuple, Dict, Callable

class EasyICDDataset(Dataset):
	"""
	EasyICDDataset: 
	Class to represent pytorch datasets created from EasyICD-scraped
	image datasets. Depends on the file structure and naming conventions followed
	by the easy_icd.scraping.scrape_images.scrape_images function to avoid having to
	load image path lists and labels, and so will not work with arbitrary image
	datasets.
	"""
	def __init__(self, image_dir: str, class_names: List[str],
				 image_transform: Optional[Callable] = None,
				 target_transform: Optional[Callable] = None,
				 scale_images: Optional[bool] = False,
				 image_indices: Optional[List[np.ndarray]] = None) -> None:
		"""
		Constructor for EasyICD dataset objects.
		"""
		self.image_dir = image_dir
		self.class_dirs = [os.path.join(
			image_dir, class_name.replace(' ', '_')) for class_name in class_names]

		self.class_names = class_names
		self.num_images_per_class = []
		self.image_inds = []

		if image_indices is None:
			for i in range(len(class_names)):
				class_scraping_info_raw = open(os.path.join(
					self.class_dirs[i], 'class_scraping_info.json')).read()

				class_scraping_info = json.loads(scraping_info_raw)

				self.num_images_per_class.append(
					class_scraping_info['num_saved_images'])

				self.image_inds.append(np.arange(self.num_images_per_class[-1]))
		else:
			self.image_inds = image_indices
			self.num_images_per_class = [len(image_indices[i])
				for i in range(len(image_indices))]
		
		self.cum_num_images_per_class = np.cumsum([0] + self.num_images_per_class)

		self.label_to_class_mapping = {
			i: self.class_names[i] for i in range(len(self.class_names))}
		
		self.scale_images = scale_images
		self.image_transform = image_transform
		self.target_transform = target_transform

		if self.scale_images:
			scaler_func = lambda x: torch.divide(x, 255)
			self.scaler = Lambda(scaler_func)  

	def __len__(self) -> int:
		"""
		Get the length of the dataset.
		"""
		return self.cum_num_images_per_class[-1]

	def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
		"""
		Get an image - label pair.
		"""
		label = np.min(np.argwhere(
			self.cum_num_images_per_class > idx).flatten()) - 1

		base_image_count = self.cum_num_images_per_class[label]
		image_number = self.image_inds[label][idx - base_image_count]

		image_path = os.path.join(self.class_dirs[label],
			f'{image_number}.jpg')
		
		image = read_image(image_path)

		if self.scale_images:
			image = self.scaler(image)

		if self.image_transform:
			image = self.image_transform(image)
			
		if self.target_transform:
			label = self.target_transform(label)
			
		return image, label
	
	def get_label_to_class_mapping(self) -> Dict:
		"""
		Get label to class mapping:
		Get the mapping from numeric labels to class names.
		"""
		return self.label_to_class_mapping		


def get_one_hot_transform(num_classes: int) -> torch.Tensor:
	"""
	Get one hot transform:
	Get a transform that converts numeric labels into one-hot encoded vectors.
	"""
	def one_hot_transform(label: int) -> torch.Tensor:
		"""
		One hot transform:
		Transform a numeric label into a one-hot vector.
		"""
		one_hot_vector = torch.zeros(num_classes)
		one_hot_vector[label] = 1
	
		return one_hot_vector
	
	return one_hot_transform
	

def create_dataset(image_dir: str, class_names: Optional[List[str]] = None,
				   one_hot_labels: Optional[bool] = False,
				   scale_images: Optional[bool] = False,
				   train_test_split_ratio: Optional[float] = 1.0) -> EasyICDDataset:
	"""
	Create dataset:
	Create an EasyICD dataset from the images scraped using easy_icd.scraping
	functions.
	"""
	if class_names is None:
		class_names = [i for i in os.listdir(image_dir) if os.path.isdir(
			os.path.join(image_dir, i))]

	num_classes = len(class_names)
		
	if one_hot_labels:
		target_transform = get_one_hot_transform(num_classes)
	else:
		target_transform = None
		
	image_transform = None
	
	if train_test_split_ratio < 1.0:
		train_inds = []
		test_inds = []

		for i in range(num_classes):
			class_scraping_info_raw = open(os.path.join(
				image_dir, class_names[i], 'class_scraping_info.json')).read()

			class_scraping_info = json.loads(class_scraping_info_raw)
			num_images_in_class = class_scraping_info['num_saved_images']
			num_train_images = int(train_test_split_ratio * num_images_in_class)

			sel_inds = np.zeros(num_images_in_class).astype(bool)
			sel_inds[np.random.choice(num_images_in_class,
				num_train_images, False)] = True

			train_inds.append(np.argwhere(sel_inds).flatten())
			test_inds.append(np.argwhere(np.invert(sel_inds)).flatten())		

		train_dataset = EasyICDDataset(image_dir, class_names, image_transform,
			target_transform, scale_images, train_inds)

		test_dataset = EasyICDDataset(image_dir, class_names, image_transform,
			target_transform, scale_images, test_inds)

		return train_dataset, test_dataset
	else:
		dataset = EasyICDDataset(image_dir, class_names, image_transform,
			target_transform, scale_images)
	
		return dataset


def compute_dataset_stats(dataloader: DataLoader, num_minibatches: Optional[int] = 250):
	"""
	Estimate the per-channel means and stds of a dataset.

	Args:
		dataloader: DataLoader that fetches images from the dataset.
		num_minibatches: number of minibatches to aggregate statistics over.
	"""
	means = torch.zeros((num_minibatches, 3))
	stds = torch.zeros((num_minibatches, 3))

	for i in range(num_minibatches):
		images, labels = next(iter(dataloader))

		curr_means = torch.mean(images, (0, 2, 3))
		curr_stds = torch.std(images, (0, 2, 3))

		means[i] = curr_means
		stds[i] = curr_stds

	mean = torch.mean(means, 0)
	std = torch.mean(stds, 0)

	return mean, std