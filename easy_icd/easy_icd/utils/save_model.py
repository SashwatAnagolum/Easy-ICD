"""
Author: Sashwat Anagolum
"""

import torch
import torch.nn as nn

def save_model(model, save_dir, file_name):
	"""
	Save the model.

	Args:
		model (nn.Module): the model to save.
		save_dir (str): the folder to save the model in.
		file_name (str): the name of the file to save the model state in.
	"""
	if not os.path.exists(save_dir):
		os.mkdirs(save_dir)

	torch.save(model.state_dict(), os.path.join(save_dir, file_name))