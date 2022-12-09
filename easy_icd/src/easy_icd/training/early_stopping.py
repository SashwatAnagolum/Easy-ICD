"""
Author: Sashwat Anagolum
"""

import numpy as np

from typing import Optional

class EarlyStopper():
	"""
	Convenience class to check for early stopping based on the loss incurred on a 
	held-out portion of the dataset.
	"""
	def __init__(self, patience: Optional[int] = 5):
		"""
		Constructor for objects of class EarlyStopper.

		Args:
			patience (int): how many steps to wait after the last reduction in 
				test loss before stopping training. Defaults to 5.
		"""
		self.steps_since_last_reduction = 0
		self.loss_ema = None
		self.patience = patience

	def check_for_early_stop(self, curr_loss: float):
		"""
		Check if we need to early stop now based on the current loss passed in.

		Args:
			curr_loss (float): the current validation / test loss of the model.

		Returns:
			(bool): whether to stop training or not.
		"""
		if (self.loss_ema is None) or (self.loss_ema >= curr_loss):
			self.steps_since_last_reduction = 0
		else:
			self.steps_since_last_reduction += 1

		if self.loss_ema is None:
			self.loss_ema = curr_loss
		else:
			self.loss_ema = self.loss_ema * 0.9 + curr_loss * 0.1

		return self.steps_since_last_reduction > self.patience