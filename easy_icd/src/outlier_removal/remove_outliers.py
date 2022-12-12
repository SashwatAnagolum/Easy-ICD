"""
Author: Sashwat Anagolum
"""

import numpy as np
import os
import json

from typing import List, Optional

def remove_outliers(data_dir: str, class_names: List[str],
					num_desired_images: int, file_name: Optional[str] = '') -> None:
	"""
	Remove outliers in each class by marking them as invalid.

	Args:
		data_dir (str): folder where all the images are stored.
		class_names (list): names of the classes to remove outliers for.
		num_desired_images (int): the number of images to be kept after
			outlier removal.
		file_name (str): the name of the file to save containing the invalid image
			indices.
	"""
	class_names = [class_name.replace(' ', '_') for class_name in class_names]

	if file_name != '':
		file_name = '_' + file_name

	for idx, class_name in enumerate(class_names):
		class_dir = os.path.join(data_dir, class_name)

		num_samples = json.loads(open(os.path.join(
			class_dir, 'class_scraping_info.json'),
			'r').read())['num_saved_images']

		num_images_to_eliminate = num_samples - num_desired_images[idx]

		if num_images_to_eliminate == 0:
			continue

		hardness_scores = np.load(os.path.join(class_dir, 'hardness_scores.npy'))
		proximity_scores = np.load(os.path.join(class_dir, 'proximity_scores.npy'))
		redundancy_scores = np.load(os.path.join(class_dir, 'redundancy_scores.npy'))

		composite_scores = np.multiply(np.multiply(
			redundancy_scores, proximity_scores), hardness_scores)

		elimination_inds = np.argsort(composite_scores)[:num_images_to_eliminate]

		np.save(os.path.join(class_dir, f'invalid_inds{file_name}.npy'),
			elimination_inds)