import numpy as np
import os

from typing import List, Optional

def remove_outliers(data_dir: str, class_names: List[str],
					num_desired_images: int, file_name: Optional[str] = '') -> None:
	"""
	Remove outliers in each class by marking them as invalid.

	Args:
		data_dir: folder where all the images are stored.
		class_names: names of the classes to remove outliers for.
		num_desired_images: int rperesenting the number of images to be kept after
			outlier removal.
		file_name: the name of the file to save containing the invalid image indices.
	"""
	class_names = [class_name.replace(' ', '_') for class_name in class_names]

	if file_name is not '':
		file_name = '_' + file_name

	for idx, class_name in enumerate(class_names):
		class_dir = os.path.join(data_dir, class_name)

		hardness_scores = np.load(os.path.join(class_dir, 'hardness_scores.npy'))
		proximity_scores = np.load(os.path.join(class_dir, 'proximity_scores.npy'))
		redundancy_scores = np.load(os.path.join(class_dir, 'redundancy_scores.npy'))

		num_samples = hardness_scores.shape[0]
		num_images_to_eliminate = num_samples - num_desired_images[idx]

		if num_images_to_eliminate == 0:
			continue
		else:
			proximity_elimination = num_images_to_eliminate // 2
			redundancy_elimination = (num_images_to_eliminate - proximity_elimination)
			redundancy_elimination = 2 * redundancy_elimination // 3
			hardness_elimination = num_images_to_eliminate - (
				proximity_elimination + redundancy_elimination)

			elimination_inds = []

			proximity_ranks = np.argsort(proximity_scores)
			elimination_inds.append(proximity_ranks[:proximity_elimination])

			remaining = proximity_ranks[proximity_elimination:]

			redundancy_ranks = remaining[np.argsort(redundancy_scores[remaining])]
			elimination_inds.append(redundancy_ranks[::-1][:redundancy_elimination])

			remaining = redundancy_ranks[::-1][redundancy_elimination:]

			hardness_ranks = remaining[np.argsort(hardness_scores[remaining])]
			elimination_inds.append(hardness_ranks[::-1][:hardness_elimination])

			np.save(os.path.join(class_dir, f'invalid_inds{file_name}.npy'),
				np.concatenate(elimination_inds))