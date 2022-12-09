"""
Author: Sashwat Anagolum
"""

import numpy as np
import torch
import torch.nn as nn

from typing import Optional, Tuple, List, Union

class LinearClassifier(nn.Module):
	"""
	Build a linear classifer using the features learnt by another supervised 
	model.
	"""
	def __init__(self, trained_model: torch.nn.Module, in_size: int,
				 num_classes: int):
		"""
		Constructor for objects of class LinearClassifier.

		Args:
			trained_model (nn.Module): the trained feature extractor.
			in_size (int): output size of the feature extractor.
			num_class (int): number of classes in the dataset to be learnt.
		"""
		super(LinearClassifier, self).__init__()

		self.layer = nn.Linear(in_size, num_classes)
		self.trained_model = trained_model
		self.trained_model.eval()
		self.trained_model.use_projection_head(False)
		self.trained_model.requires_grad_(False)

	def finetune(self, finetune_or_not: bool):
		"""
		Whether to finetune the entire model or not.

		Args:
			finetune_or_not (bool): whether to finetune or linear probe.
		"""
		self.trained_model.requires_grad_(finetune_or_not)

		if finetune_or_not:
			self.trained_model.train()
		else:
			self.trained_model.eval()

	def forward(self, x):
		"""
		Forward pass for the linear classifier. First extracts features using the 
			trained model, and then performs classification based on those features.

		Args:
			x (torch.Tensor): input data.

		Returns:
			(torch.Tensor): class logits.
		"""
		features = self.trained_model(x)

		return self.layer(features)


class ResNetBasicBlock(nn.Module):
	"""
	Basic ResNet block, including convolutions, batch norm, and residual connections.
	"""
	def __init__(self, in_channels: int, out_channels: int, kernel_stride: int,
				 add_act: Optional[bool] = True):
		"""
		Constructor for objects of class ResNetBasicBlock.

		Args:
			in_channels (int): number of input channels.
			out_channels (int): number of output channels.
			kernel_stride (int): kernel stride length.
			add_act (bool, optional): whether to add activations at the end of the block
				or not. Defaults to True.
		"""
		super(ResNetBasicBlock, self).__init__()

		self.layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3,
				stride=kernel_stride, padding=1, bias=False),
			nn.BatchNorm2d(out_channels), nn.ReLU(),
			nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1,
				padding='same', bias=False),
			nn.BatchNorm2d(out_channels)]

		if add_act:
			self.layers.append(nn.ReLU())

		self.layers = nn.Sequential(*self.layers)

		self.shortcut = nn.Sequential()

		if in_channels != out_channels or kernel_stride != 1:
			self.shortcut = nn.Sequential(
				nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=kernel_stride,
					bias=False),
				nn.BatchNorm2d(out_channels)
			)

	def forward(self, x: torch.Tensor):
		"""
		Forward pass of the block.

		Args:
			x (torch.Tensor): input data.

		Returns:
			(torch.Tensor): block output.     
		"""
		out = self.layers(x)
		out = out + self.shortcut(x)

		return out


class ResNet(nn.Module):
	"""
	Build ResNet-based models.
	"""
	def __init__(self, num_layers: Optional[int] = 4,
				 num_blocks: Optional[List[int]] = None,
				 in_channels: Optional[int] = 3,
				 out_channels: Optional[List[int]] = None,
				 linear_sizes: Optional[List[int]] = None,
				 supervised: Optional[bool] = False):
		"""
		Constructor for objects of class ResNetOutlierDetectionModel.

		Args:
			num_layers (int, optional): number of resnet layers to use. Can only be
				3 or 4. Defaults to 4.
			num_blocks (list, optional): the number of ResNet blocks in each
				of the four layers to use. Defaults to [1, 1, 1, 1].
			in_channels (int, optional): int indicating the number of input channel
				in the data to be learnt. Defaults to 3.
			out_channels (list, optional): List of ints indicating the number of
				output channels for each of the four ResNet layers. Defaults
				to [16, 32, 64, 128].
			supervised (bool, optional): bool indicating whether the model will be used
				for supervised learning or not. 
		"""
		super(ResNet, self).__init__()

		if num_blocks is None:
			num_blocks = [2, 2, 2, 2]

		if out_channels is None:
			out_channels = [16, 32, 64, 128]

		if linear_sizes is None:
			linear_sizes = [256, 64]

		self.stem = nn.Sequential(
			nn.Conv2d(in_channels, out_channels[0], kernel_size=3, stride=1,
				padding='same', bias=False),
			nn.BatchNorm2d(out_channels[0]),
			nn.ReLU()
		)

		self.layer1 = self.make_resnet_layer(0, out_channels[0], out_channels[0],
			num_blocks[0], 1)

		self.layer2 = self.make_resnet_layer(1, out_channels[0], out_channels[1],
			num_blocks[1], 2)

		self.layer3 = self.make_resnet_layer(2, out_channels[1], out_channels[2],
						num_blocks[2], 2)

		if num_layers == 3:
			self.layer4 = nn.Identity()
		else:
			if num_layers != 4:
				print('Only 3 or 4 layers currently supported! Current choice for',
					f'number of layers {num_layers} is invalid. Defaulting to 4 layers.')

			self.layer4 = self.make_resnet_layer(3, out_channels[2], out_channels[3],
				num_blocks[3], 2)

		self.avg_pool = nn.AdaptiveAvgPool2d((2, 1))

		if not supervised:
			self.linear_layers = nn.Identity()

			self.projection_head = nn.Sequential(
				nn.Linear(2 * out_channels[num_layers - 1], linear_sizes[0]),
				nn.ReLU(),
				nn.Linear(linear_sizes[0], linear_sizes[1])
			)
		else:
			linear_sizes.insert(0, 2 * out_channels[num_layers - 1])
			self.linear_layers = self.make_linear_layers(
				linear_sizes[:-1])

			self.projection_head = nn.Linear(linear_sizes[-2],
				linear_sizes[-1])

		self._use_projection_head = True

		for module in self.modules():
			if isinstance(module, nn.Conv2d):
				nn.init.kaiming_normal_(module.weight, mode='fan_out',
					nonlinearity='relu')
			elif isinstance(module, nn.Linear):
				nn.init.kaiming_normal_(module.weight, mode='fan_out',
					nonlinearity='relu')

				nn.init.constant_(module.bias, 0)				
			elif isinstance(module, nn.BatchNorm2d):
				nn.init.constant_(module.weight, 1)
				nn.init.constant_(module.bias, 0)

	def make_resnet_layer(self, layer_num: int, in_channels: int, out_channels: int,
						  num_blocks: int, stride: int) -> torch.nn.Module:
		""" 
		Make a ResNet layer using multiple ResNet blocks.

		Args:
			layer_num (int): the current layer being built.
			in_channels (int): the number of input channels.
			out_channels (int): the number of output channels.
			num_blocks (int): the number of basic blocks to be used
				to construct the layer.

		Returns:
			(nn.Sequential): built resnet layer.
		"""
		strides = [stride] + [1] * (num_blocks - 1)
		layer_blocks = []

		for i in range(num_blocks):
			add_act = (i != (num_blocks - 1)) or (layer_num != 3)

			if i == 0:
				layer_blocks.append(ResNetBasicBlock(in_channels, out_channels,
					strides[i], add_act))				
			else:
				layer_blocks.append(ResNetBasicBlock(out_channels, out_channels,
					strides[i], add_act)) 

		return nn.Sequential(*layer_blocks) 

	def make_linear_layers(self, linear_layer_sizes):
		"""
		Construct linear layers for the model.

		Args:
			linear_layer_sizes (list): list of linear layer input and output sizes.

		Returns:
			(nn.Sequential): list of linear layers.
		"""
		linear_layers = []

		for i in range(len(linear_layer_sizes) - 1):
			linear_layers.append(nn.Linear(linear_layer_sizes[i],
				linear_layer_sizes[i + 1]))

			if i != (len(linear_layer_sizes) - 2):
				linear_layers.append(nn.ReLU())

		return nn.Sequential(*linear_layers)

	def use_projection_head(self, use_or_not: bool):
		"""
		Whether to use the projection head or not. The projection head is used during
		training, but discarded afterwards for linear evaluation and other downstream
		tasks using the trained model.

		Args:
			use_or_not (bool): whether to use the projection head or not.
		"""
		self._use_projection_head = use_or_not

	def forward(self, x: torch.Tensor):
		"""
		Forward pass for the model.

		Args:
			x (torch.Tensor): input data.

		Returns:
			(Torch.tensor): learned features.     
		"""
		out = self.stem(x)
		out = self.layer1(out)
		out = self.layer2(out)
		out = self.layer3(out)
		out = self.layer4(out)
		out = self.avg_pool(out)
		out = torch.flatten(out, 1)
		out = self.linear_layers(out)

		if self._use_projection_head:
			out = self.projection_head(out)

		return out	
		