import numpy as np
import torch
import torch.nn as nn

"""
DeeperDeepSEA model architecture with 5 shape features and min-max normalization.
Network takes in two inputs: sequence (first four) and shape (remainder) features.
Each input is passed through two convolutional layers before being merged.
"""


class DeeperDeepSEA(nn.Module):
	def __init__(self, sequence_length, n_targets):
		super(DeeperDeepSEA, self).__init__()
		conv_kernel_size = 4
		pool_kernel_size = 4

		# Provide two inputs to the network: sequence and shape features
		self.conv_net_seq = nn.Sequential(
			nn.Conv1d(4, 450, kernel_size=conv_kernel_size),  # 4 channels for sequence
			nn.ReLU(inplace=True),
			nn.Conv1d(450, 450, kernel_size=conv_kernel_size),  
			nn.ReLU(inplace=True)
		)

		self.conv_net_shape = nn.Sequential(
			nn.Conv1d(5, 450, kernel_size=conv_kernel_size),  # 5 channels for shape features
			nn.ReLU(inplace=True),
			nn.Conv1d(450, 450, kernel_size=conv_kernel_size),
			nn.ReLU(inplace=True)
		)

		self.conv_net = nn.Sequential(  # Merge the two inputs
			nn.MaxPool1d(kernel_size=pool_kernel_size, stride=pool_kernel_size),
			nn.BatchNorm1d(900),
			nn.Conv1d(900, 450, kernel_size=conv_kernel_size),
			nn.ReLU(inplace=True),
			nn.Conv1d(450, 450, kernel_size=conv_kernel_size),
			nn.ReLU(inplace=True),
			nn.MaxPool1d(kernel_size=pool_kernel_size, stride=pool_kernel_size),
			nn.BatchNorm1d(450),
			nn.Dropout(p=0.2),
			nn.Conv1d(450, 450, kernel_size=conv_kernel_size),
			nn.ReLU(inplace=True),
			nn.Conv1d(450, 450, kernel_size=conv_kernel_size),
			nn.ReLU(inplace=True),
			nn.BatchNorm1d(450),
			nn.Dropout(p=0.2)
		)

		# Calculate the number of channels in the final layer
		reduce_by = 2 * (conv_kernel_size - 1)  # Each conv layer reduces the seq length by kernel_size - 1
		pool_kernel_size = float(pool_kernel_size)  #
		self._n_channels = int(  # Number of channels in the final layer
			np.floor(
				(np.floor(
					(np.floor(
						(sequence_length - reduce_by) / pool_kernel_size)
					 - reduce_by) / pool_kernel_size)
				 - reduce_by)
			)
		)

		self.classifier = nn.Sequential(  # Final classification step
			nn.Linear(450 * self._n_channels, n_targets),  # Linear layer
			nn.ReLU(inplace=True),  # ReLU activation
			nn.BatchNorm1d(n_targets),  # Batch normalization
			nn.Linear(n_targets, n_targets),  # Linear layer
			nn.Sigmoid()  # Sigmoid activation
		)

	def fmin(self, x):
		x, _ = x.min(dim=0, keepdim=True)
		x, _ = x.min(dim=2, keepdim=True)
		return x

	def fmax(self, x):
		x, _ = x.max(dim=0, keepdim=True)
		x, _ = x.max(dim=2, keepdim=True)
		return x

	def normalize(self, x, epsilon=1e-7):
		return (x - self.fmin(x)) / (self.fmax(x) - self.fmin(x) + epsilon)

	def forward(self, x):
		seq, shape = x[:, :4], x[:,
		                       4:]  # splits the input tensor into two parts: seq (sequence features, first four) and shape (shape features, remainder).
		seq = self.normalize(seq)
		shape = self.normalize(shape)
		out_seq = self.conv_net_seq(seq)
		out_shape = self.conv_net_shape(shape)
		out = torch.cat((out_seq, out_shape), dim=1)  # concatenates sequence and shape feature maps.
		out = self.conv_net(out)  # pass the concatenated features through the main convolutional network.
		reshape_out = out.view(out.size(0),
		                       450 * self._n_channels)  # reshaping the output tensor for input into the classifier.
		predict = self.classifier(reshape_out)  # final classification step
		return predict


def criterion():
	"""
	Specify the appropriate loss function (criterion) for this
	model.

	Returns
	-------
	torch.nn._Loss
	"""
	return nn.BCELoss()


def get_optimizer(lr):
	"""
	Specify an optimizer and its parameters.

	Returns
	-------
	tuple(torch.optim.Optimizer, dict)
		The optimizer class and the dictionary of kwargs that should
		be passed in to the optimizer constructor.
		"""
	return (torch.optim.SGD,
	        {"lr": lr, "weight_decay": 1e-6, "momentum": 0.9})
