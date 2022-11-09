import torch
import numpy as np
import os
import json

from torch.utils.data import Dataset, DataLoader, RandomSampler
from torchvision import transforms, utils
from torchvision.io import read_image
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
				 target_transform: Optional[Callable] = None) -> None:
		"""
		Constructor for EasyICD dataset objects.
		
		Args:
		    image_dir :
		    	str - full folder path to save images to
		    class_names : 
		    	List[str] - List of keywords used in search
		    image_transform : 
		    	Optional[Callable] - image transformation method to apply to images
		    target_transform : 
		    	Optional[Callable] - method to apply transformations to masks aswell 
		"""
		self.image_dir = image_dir
		self.class_dirs = [os.path.join(
			image_dir, class_name.replace(' ', '_')) for class_name in class_names]

		self.class_names = class_names
		self.num_images_per_class = []
	
		scraping_info_raw = open(os.path.join(
			image_dir, 'scraping_info.json')).read()

		scraping_info = json.loads(scraping_info_raw)
		
		for class_name in class_names:
			self.num_images_per_class.append(
				scraping_info[class_name]['num_saved_images'])
		
		self.cum_num_images_per_class = np.cumsum([0] + self.num_images_per_class)
		
		self.label_to_class_mapping = {
			i: self.class_names[i] for i in range(len(self.class_names))}
		
		self.image_transform = image_transform
		self.target_transform = target_transform

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
		image_path = os.path.join(self.class_dirs[label],
			f'{idx - base_image_count}.jpg')
		
		image = read_image(image_path)
		
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
				   one_hot_labels: Optional[bool] = False) -> EasyICDDataset:
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
	
	dataset = EasyICDDataset(image_dir, class_names, image_transform,
		target_transform)
	
	return dataset
