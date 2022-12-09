"""
Author: Sashwat Anagolum
"""

import torch
import torch.nn as nn 

from typing import Optional

class CELoss(nn.Module):
	"""
	Standard cross-entropy loss.
	"""
	def __init__(self):
		"""
		Constructor for CELoss objects.
		"""
		super(CELoss, self).__init__()

		self.loss_fn = nn.CrossEntropyLoss(reduction='mean')

	def forward(self, predictions, labels):
		"""
		Compute the cross-entropy loss.

		Args:
			predictions (torch.Tensor): predictions made by the model.
			labels (torch.Tensor): ground truth labels.

		Returns:
		    (torch.Tensor): Average cross-entropy loss over the minibatch.
		"""
		original_batch_size = labels.shape[0]
		num_samples = predictions.shape[0]
		num_views = num_samples // original_batch_size

		labels = labels.repeat(num_views)

		return self.loss_fn(predictions, labels)


class SimCLRLoss(nn.Module):
	"""
	SimCLR loss. Can use the original formulation, or the modified version excluding
		negative samples that belong to the same class as the anchor sample.
	"""
	def __init__(self, temperature: Optional[float] = 0.07,
				 normalize_features: Optional[bool] = True,
				 reduce_mean: Optional[bool] = True,
				 use_labels: Optional[bool] = True,
				 only_pairwise_loss: Optional[bool] = False):
		"""
		Constructor for SimCLRLoss objects.

		Args:
			temperature (float, optional): temp to use with the softmax computation for
				similarities between sample representations.
			normalize_features (bool, optional): bool that determines whether to
					normalize representations before computing the loss or not.
			reduce_mean (bool, optional): whether to return the mean loss or
				the per-sample loss. Defaults to True.
			use_labels (bool, optional): whether to use labels to compute the loss or
				not. Defaults to True.
			only_pairwise_loss: whether to use negative samples to compute the loss
				or only use positive pair similarities. Defaults to False.
		"""
		super(SimCLRLoss, self).__init__()

		self.temperature = temperature
		self.normalize_features = normalize_features
		self.reduce_mean = reduce_mean
		self.use_labels = use_labels
		self.only_pairwise_loss = only_pairwise_loss

	def forward(self, features: torch.Tensor, 
				labels: Optional[torch.Tensor] = None) -> torch.Tensor:
		"""
		Compute the contrastive loss for a minibatch of learned representations. 
		If labels is None, it reduces to the SimCLR
		unsupervised loss.

		Args:
			features (torch.Tensor): Tensor of shape [batch_size, n_views, ...]
				containing the learned representations of the data. The first
				dimension indicates minibatch size, and the second dimension indicates
				the number of  augmented versions (views) of each original sample
				from the data.
			labels (torch.Tensor, optional): torch.Tensor of shape [batch_size]
				containing the labels for the minibatch. Defaults to None. If provided,
				the loss will be computed by comparing similarities between different
				views of each image, and the similarities of each of the views with
				all other image views that are not derived from images with the same
				class label.

		Returns:
			(torch.Tensor): The mean constrastive loss for the minibatch.
		"""
		device = torch.device('cuda' if features.is_cuda else 'cpu')

		batch_size = labels.shape[0]
		num_samples = features.shape[0]
		num_views = num_samples // batch_size

		features = torch.stack(torch.split(features,
			[batch_size for i in range(num_views)], 0), 1)

		if self.normalize_features:
			features = torch.div(features, torch.linalg.norm(features, dim=2,
				keepdim=True))
	
		flattened_features = torch.cat(torch.unbind(features, dim=1), dim=0)
		feature_similarities = torch.matmul(flattened_features, flattened_features.T)

		no_self_similarity_mask = torch.ones((num_samples, num_samples),
			dtype=torch.float32).to(device)

		no_self_similarity_mask = torch.sub(no_self_similarity_mask,
			torch.eye(num_samples, dtype=torch.float32).to(device))

		numerator_mask = torch.eye(batch_size, dtype=torch.float32).to(device)
		numerator_mask = torch.mul(numerator_mask.repeat(num_views, num_views),
			no_self_similarity_mask)

		feature_similarities = torch.div(feature_similarities, self.temperature)
		max_similarities = torch.max(feature_similarities, dim=1,
			keepdim=True)[0].detach()

		feature_similarities = torch.sub(feature_similarities, max_similarities)
		all_similarities = torch.mul(torch.exp(feature_similarities),
			no_self_similarity_mask)

		if not self.only_pairwise_loss:
			numerator_similarities = torch.mul(numerator_mask, all_similarities).sum(1)

			denominator_mask = torch.ones((num_samples, num_samples)).to(device)
			denominator_mask = torch.mul(denominator_mask, no_self_similarity_mask)

			if (labels is not None) and self.use_labels:
				labels = labels.view(-1, 1)
				labels_same_filter = torch.eq(labels, labels.T).float().to(
					device).repeat(num_views, num_views)

				labels_same_filter = torch.sub(labels_same_filter, numerator_mask)
				denominator_mask = torch.sub(denominator_mask, labels_same_filter)

			denominator_similarities = torch.mul(denominator_mask,
				all_similarities).sum(1)

			view_softmax_probs = torch.div(numerator_similarities,
				denominator_similarities)

			loss = torch.mul(torch.log(view_softmax_probs), -1)
		else:
			similarities = torch.mul(numerator_mask, all_similarities).sum(1)
			loss = torch.mul(torch.log(similarities), -1)

		if self.reduce_mean:
			loss = torch.mean(loss)

		return loss
		