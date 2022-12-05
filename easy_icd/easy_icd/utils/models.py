import numpy as np
import torch
import torch.nn as nn

from typing import Optional, Tuple, List, Union

class LinearClassifier(torch.nn.Module):
	"""
	Build a linear classifer using the features learnt by another supervised 
	model.
	"""
	def __init__(self, trained_model: torch.nn.Module, in_size: int,
				 num_classes: int):
		"""
		Constructor for objects of class LinearClassifier.
		"""
		super(LinearClassifier, self).__init__()

		self.layer = nn.Linear(in_size, num_classes)
		self.trained_model = trained_model
		self.trained_model.eval()
		self.trained_model.use_projection_head(False)
		self.trained_model.requires_grad_(False)

	def finetune(self, finetune_or_not: bool):
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
			x: input data.

		Returns:
			Torch.tensor of class probabilities.
		"""
		features = self.trained_model(x)

		return self.layer(features)


class ResNetBasicBlock(torch.nn.Module):
	"""
	Basic ResNet block, including convolutions, batch norm, and residual connections.
	"""
	def __init__(self, in_channels: int, out_channels: int, kernel_stride: int,
				 add_act: Optional[bool] = True):
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
			x: torch.Tensor of input data.

		Returns:
			torch.Tensor of model output.       
		"""
		out = self.layers(x)
		out = out + self.shortcut(x)

		return out


class ResNet(torch.nn.Module):
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
			num_layers: int indicating the number of resnet layers to use. Can only be
				3 or 4. Defaults to 4.
			num_blocks: List of ints indicating the number of ResNet blocks in each
				of the four layers to use. Defaults to [1, 1, 1, 1].
			in_channels: int indicating the number of input channels in the data to be
				learnt. Defaults to 3.
			out_channels: List of ints indicating the number of output channels for 
				each of the four ResNet layers. Defaults to [16, 32, 64, 128].
			supervised: bool indicating whether the model will be used for supervised
				learning or not. 
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

			print(self.projection_head)

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
			layer_num: int index of the current layer being built.
			in_channels: int indicating the number of input channels.
			out_channels: int indicating the number of output channels.
			num_blocks: int indicating the number of basic blocks to be used
				to construct the layer.
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
		linear_layers = []

		for i in range(len(linear_layer_sizes) - 1):
			print(linear_layer_sizes[i], linear_layer_sizes[i + 1])
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
			use_or_not: bool indicating whether to use the projection head or not.
		"""
		self._use_projection_head = use_or_not

	def forward(self, x: torch.Tensor):
		"""
		Forward pass of the model.

		Args:
			x: torch.Tensor of input data.

		Returns:
			torch.Tensor of model output.       
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


class SimpleOutlierDetectionModel(torch.nn.Module):
	"""
	Build pytorch models for outlier detection.
	"""
	def __init__(self, image_size: int, flattened_size: int, 
				 num_blocks: Optional[int] = 4,
				 num_conv_layers_per_block: Optional[int] = 1, 
				 num_conv_filters: Optional[Union[int, List[int], np.ndarray]] = None,
				 pooling_ratios: Optional[Union[int, List[int], np.ndarray]] = None,
				 linear_layer_sizes: Optional[Union[List[int], np.ndarray]] = None,
				 use_stem: Optional[bool] = True, stem_stride_size: Optional[int] = 4,
				 stem_num_filters: Optional[int] = 16):
		"""
		Constructor for OutlierDetectionModel objects.

		Args:
			image_size: int indicating the input image dimension.
			flattened_size: int indicating the size of the inputs to the first
				linear layer of the model.
			num_blocks: int representing the number of Conv / Pool blocks
				to use in the network. Defaults to 4.
			num_conv_layers_per_block: int representing the number of conv
				layers to use per blck. Defaults to 1.
			num_conv_filters: int / Array-like of ints representing the number
				of conv filters to be used in each block. Defaults to 
				16 * (2 ** block_number).
			pooling_ratios: int / Array-like of ints representing the downsampling
				factor for each block. Defaults to 2 for all blocks.
			linear_layer_sizes: Array-like of ints representing the sizes of
				the hidden layers to be used after all the conv / pool blocks.
				Defaults to [128, 128].
			use_stem: bool indicating whether to use a stem Conv2d layer to reduce
				image size or not.
			stem_stride_size: int representing the size of the stride used for the
				stem Conv2d layer. The size of the kernel is set to the same size as
				the stride.
			stem_num_filters: int representing the number of filters used by the stem
				layer.
		"""
		super(OutlierDetectionModel, self).__init__()

		self.embedding_mode = False

		self.image_size = image_size
		self.num_blocks = num_blocks
		self.num_conv_layers_per_block = num_conv_layers_per_block
		self.use_stem = use_stem
		self.stem_stride_size = stem_stride_size
		self.stem_num_filters = stem_num_filters
		self.flattened_size = flattened_size

		# need to add error checking because we use grouped conv layers
		if num_conv_filters is None:
			self.num_conv_filters = [16 * (2 ** i) for i in range(self.num_blocks)]
		else:
			if isinstance(num_conv_filters, int):
				self.num_conv_filters = [num_conv_filters for i in range(
					self.num_blocks)]
			else:
				self.num_conv_filters = num_conv_filters

		if self.use_stem:
			self.num_conv_filters.insert(0, self.stem_num_filters)
		else:
			self.num_conv_filters.insert(0, 3)

		if pooling_ratios is None:
			self.pooling_ratios = [2 for i in range(self.num_blocks)]
		else:
			if isinstance(pooling_ratios, int):
				self.pooling_ratios = [pooling_ratios for i in range(self.num_blocks)]
			else:
				self.pooling_ratios = pooling_ratios

		if linear_layer_sizes is None:
			self.linear_layer_sizes = [128, 128]
		else:
			self.linear_layer_sizes = linear_layer_sizes

		self.num_linear_layers = len(self.linear_layer_sizes)
		self.linear_layer_sizes.insert(0, self.flattened_size)

		# need to convert all of these into input parameters
		# some of these are not implemented yet
		self.conv_layers_kernel_size = 3
		self.use_residual_connections = False # not implemented
		self.use_skip_connections = True # not implemented
		self.pooling_type = 'max'
		self.dropout_prob = 0.2 # not implemented

		self.build_model()

	def build_model(self):
		self.layers = []

		# append the downsampling 'patchify' stem layer
		if self.use_stem:
			self.layers.append(nn.ZeroPad2d(3))

			self.layers.append(nn.Conv2d(3, self.stem_num_filters,
				self.stem_stride_size, self.stem_stride_size, 'valid'))

			self.layers.append(nn.ReLU())

		for i in range(self.num_blocks):
			# regular conv layer to process inputs across channels
			self.layers.append(nn.Conv2d(self.num_conv_filters[i],
				self.num_conv_filters[i + 1],
				self.conv_layers_kernel_size, 1, 'same'))   
			self.layers.append(nn.ReLU())   

			input_channels = self.num_conv_filters[i + 1]   

			for j in range(self.num_conv_layers_per_block - 1):
				self.layers.append(nn.Conv2d(input_channels, input_channels,
					self.conv_layers_kernel_size, 1, 'same')) 
				
			if self.num_conv_layers_per_block > 1:
				self.layers.append(nn.ReLU())

			# downsampling via pooling
			if self.pooling_type == 'max':
				self.layers.append(nn.MaxPool2d(self.pooling_ratios[i]))
			elif self.pooling_type == 'avg':
				self.layers.append(nn.AvgPool2d(self.pooling_ratios[i])) 
				
		# add flatten layer to flatten conv outputs
		self.layers.append(nn.Flatten())

		# add linear layers to model
		for i in range(self.num_linear_layers):
			self.layers.append(nn.Linear(self.linear_layer_sizes[i],
				self.linear_layer_sizes[i + 1]))

			if i != self.num_linear_layers - 1:
				self.layers.append(nn.ReLU())

		for layer in self.layers:
			if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
				nn.init.kaiming_normal_(layer.weight,
					mode='fan_out', nonlinearity='relu')
				nn.init.zeros_(layer.bias)

		self.layers = nn.ModuleList(self.layers)

	def set_embedding_mode(embedding_mode: Optional[bool] = None):
		"""
		Set the embedding mode (training / embedding) for the model.
		If embedding mode is true, then the model forward pass excludes the last
		ReLU and Linear layer. 

		Args:
			embedding_mode: bool indicating whether the model is being used in
				embedding mode or not. Defaults to toggling the current value of
				self.embedding_mode.
		"""
		if embedding_mode is None:
			self.embedding_mode = not self.embedding_mode
		else:
			self.embedding_mode = embedding_mode

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		"""
		Forward pass of the model.

		Args:
			x: input data.

		Returns:
			Torch.tensor of model output.
		""" 
		for i in range(len(self.layers) - 2):
			x = self.layers[i](x)

		if not self.embedding_mode:
			x = self.layers[-1](self.layers[-2](x))

		return x


# def build_outlier_detector(image_size: int,
#                          num_images: int, num_classes: int) -> OutlierDetectionModel:
#   """
#   Build a outlier detection model based on the size and characteristics of the dataset
#   scraped by the user.
#   """
#   log_size = np.log2(image_size)

#   if (log_size % 1) < 1e-20:
#       size_is_power_of_2 = True
#   else:
#       size_is_power_of_2 = False



# image_size = 64
# num_blocks = 6
# num_conv_layers_per_block = 4
# num_conv_filters = [32, 64, 128, 64, 32, 16]
# pooling_ratios = [2, 2, 2, 1, 1, 1]
# linear_layer_sizes = [128, 128]
# use_stem = True
# stem_stride_size = 2
# stem_num_filters = 16

# flattened_size = image_size
# flattened_size //= stem_stride_size

# for i in range(len(pooling_ratios)):
#     flattened_size //= pooling_ratios[i]
	
# flattened_size **= 2
# flattened_size *= num_conv_filters[-1]