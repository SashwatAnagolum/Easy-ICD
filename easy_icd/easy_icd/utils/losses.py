import torch
import torch.nn as nn 

from typing import Optional

class SupConLoss(nn.Module):
	"""
	Supervised Contrastive Loss: https://arxiv.org/pdf/2004.11362.pdf.
	SimCLR loss: https://arxiv.org/abs/2002.05709.
	
	Code based on: https://github.com/HobbitLong/SupContrast/blob/master/losses.py
	"""
	def __init__(self, temperature: Optional[float] = 0.1,
				 contrast_mode: Optional[str] = 'all',
				 normalize_features: Optional[bool] = False):
		"""
		Constructor for SupConLoss objects.

		Args:
			temperature: float to use with the softmax computation for
				similarities between sample representations.
			contrast_mode: 'all', or 'one', determines whether to use all views or
				just one view from each sample as anchor samples for the loss
				computation.
			normalize_features: bool that determines whether to normalize
				representations before computing the loss or not.
		"""
		super(SupConLoss, self).__init__()
		
		self.temperature = temperature
		
		if contrast_mode not in ['all', 'one']:
			raise ValueError(f'Invalid value {contrast_mode} for contrast mode!\
				Valid values are "one" and "all".')
		else:
			self.contrast_mode = contrast_mode
			
		self.normalize_features = normalize_features

	def forward(self, features: torch.Tensor, labels: torch.Tensor = None,
				mask: torch.Tensor = None,
				normalize_features: bool = None) -> torch.Tensor:
		"""
		Compute the contrastive loss for a minibatch of learned representations. 
		If both `labels` and `mask` are None, it reduces to the SimCLR
		unsupervised loss.

		Args:
			features: Tensor of shape [batch_size, n_views, ...] containing the
				learned representations of the data. The first dimension indicates
				minibatch size, and the second dimension indicates the number of 
				augmented versions (views) of each original sample from the data.
			labels: Tensor of shape [batch_size] containing the true labels of the
				data.
			mask: Tensor of shape [batch_size, batch_size] that indicates which
				samples are similar to each other. If sample i and sample j from the
				minibatch are similar to each other, then the element mask[i, j] == 1,
				else mask[i, j] == 0.
			normalize_features: whether to normalize the learned representations or
				not before computing the loss, overriding the normalize_features
				setting set in the constructor.

		Returns:
			The mean constrastive loss for the minibatch.
		"""
		device = torch.device('cuda') if features.is_cuda else torch.device('cpu')
		batch_size = features.shape[0]

		if len(features.shape) < 3:
			raise ValueError('`features` needs to be at least 3 dimensional.')

		if len(features.shape) > 3:
			features = features.view(features.shape[0], features.shape[1], -1)

		if labels is not None:
			labels = labels.contiguous().view(-1, 1)
	
			if labels.shape[0] != batch_size:
				raise ValueError('Num of labels does not match num of features')
	
			mask = torch.eq(labels, labels.T).float().to(device)
		else:
			if mask is None:
				mask = torch.eye(batch_size, dtype=torch.float32).to(device)
			else:
				if mask.shape != (batch_size, batch_size):
					raise ValueError(f'Current `mask` size {mask.shape} !=\
						(batch_size, batch_size)!')
					
				mask = mask.float().to(device)
				
		if normalize_features is None:
			normalize_features = self.normalize_features

		num_views = features.shape[1]
		
		if normalize_features:
			features = torch.div(features, torch.linalg.norm(features, dim=2,
								 keepdim=True))
	
		flattened_features = torch.cat(torch.unbind(features, dim=1), dim=0)

		if self.contrast_mode == 'one':
			anchor_features = features[:, 0]
			num_anchors_per_sample = 1
		elif self.contrast_mode == 'all':
			anchor_features = flattened_features
			num_anchors_per_sample = num_views
			
		feature_similarities = torch.div(torch.matmul(anchor_features,
			flattened_features.T), self.temperature)

		max_similarity, _ = torch.max(feature_similarities, dim=1, keepdim=True)
		scaled_similarities = feature_similarities - max_similarity.detach()
		
		mask = mask.repeat(num_anchors_per_sample, num_views)

		no_self_similarity_mask = torch.scatter(torch.ones_like(mask), 1,
			torch.arange(batch_size * num_anchors_per_sample).view(-1, 1).to(device),
			0)
		
		mask = torch.mul(mask, no_self_similarity_mask)
		
		exp_similarities = torch.mul(torch.exp(scaled_similarities),
			no_self_similarity_mask)

		log_prob_similarities = scaled_similarities - torch.log(exp_similarities.sum(1,
			keepdim=True))

		mean_view_similarities = torch.mul(mask,
			log_prob_similarities).sum(1) / mask.sum(1)
		
		losses = torch.mul(torch.mul(mean_view_similarities, self.temperature), -1)
		mean_loss = losses.mean()

		return mean_loss
