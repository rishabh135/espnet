from abc import ABC, abstractmethod
# https://github.com/keitakurita/Better_LSTM_PyTorch/tree/master/better_lstm

import argparse
import logging
from typing import Callable, Collection, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from espnet2.asr.tdnn import TDNN
# import warpctc_pytorch as warp_ctc
from torch.autograd import Function

from collections import Counter



from espnet2.asr.statistical_pooling import StatisticsPooling
from espnet2.asr.tdnn_xvector import Conv1d, Linear, BatchNorm1d
from espnet2.asr.lsoftmax import LSoftmaxLinear
# from speechbrain.nnet.linear import Linear
# from speechbrain.nnet.normalization import BatchNorm1d


from loss_functions import AngularPenaltySMLoss



##############################################################################################################################################################
##############################################################################################################################################################





# https://github.com/speechbrain/speechbrain/blob/develop/speechbrain/nnet/CNN.py


# import torch
# import torch.nn as nn
from torch.nn.utils.rnn import PackedSequence
from typing import *




class Xvector(torch.nn.Module):
	"""This model extracts X-vectors for speaker recognition and diarization.
	Arguments
	---------
	device : str
		Device used e.g. "cpu" or "cuda".
	activation : torch class
		A class for constructing the activation layers.
	tdnn_blocks : int
		Number of time-delay neural (TDNN) layers.
	tdnn_channels : list of ints
		Output channels for TDNN layer.
	tdnn_kernel_sizes : list of ints
		List of kernel sizes for each TDNN layer.
	tdnn_dilations : list of ints
		List of dilations for kernels in each TDNN layer.
	lin_neurons : int
		Number of neurons in linear layers.
	Example
	-------
	>>> compute_xvect = Xvector('cpu')
	>>> input_feats = torch.rand([5, 10, 40])
	>>> outputs = compute_xvect(input_feats)
	>>> outputs.shape
	torch.Size([5, 1, 512])
	"""

	def __init__(
		self,
		device="cuda",
		activation=torch.nn.LeakyReLU,
		tdnn_blocks=5,
		tdnn_channels=[512, 512, 512, 512, 1500],
		tdnn_kernel_sizes=[5, 3, 3, 1, 1],
		tdnn_dilations=[1, 2, 3, 1, 1],
		lin_neurons=512,
		in_channels=40,
	):

		super().__init__()
		self.blocks = torch.nn.ModuleList()

		# TDNN layers
		for block_index in range(tdnn_blocks):
			out_channels = tdnn_channels[block_index]
			self.blocks.extend(
				[
					Conv1d(
						in_channels=in_channels,
						out_channels=out_channels,
						kernel_size=tdnn_kernel_sizes[block_index],
						dilation=tdnn_dilations[block_index],
					),
					activation(),
					BatchNorm1d(input_size=out_channels),
				]
			)
			in_channels = tdnn_channels[block_index]

		# Statistical pooling
		self.blocks.append(StatisticsPooling())

		# Final linear transformation
		self.blocks.append(
			Linear(
				input_size=out_channels * 2,
				n_neurons=lin_neurons,
				bias=True,
				combine_dims=False,
			)
		)

	def forward(self, x, lens=None):
		"""Returns the x-vectors.
		Arguments
		---------
		x : torch.Tensor
		"""

		for layer in self.blocks:
			try:
				x = layer(x, lengths=lens)
			except TypeError:
				x = layer(x)
		return x




class VariationalDropout(torch.nn.Module):
	"""
	Applies the same dropout mask across the temporal dimension
	See https://arxiv.org/abs/1512.05287 for more details.
	Note that this is not applied to the recurrent activations in the LSTM like the above paper.
	Instead, it is applied to the inputs and outputs of the recurrent layer.
	"""
	def __init__(self, dropout: float, batch_first: Optional[bool]=False):
		super().__init__()
		self.dropout = dropout
		self.batch_first = batch_first

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		if not self.training or self.dropout <= 0.:
			return x

		is_packed = isinstance(x, PackedSequence)
		if is_packed:
			x, batch_sizes = x
			max_batch_size = int(batch_sizes[0])
		else:
			batch_sizes = None
			max_batch_size = x.size(0)

		# Drop same mask across entire sequence
		if self.batch_first:
			m = x.new_empty(max_batch_size, 1, x.size(2), requires_grad=False).bernoulli_(1 - self.dropout)
		else:
			m = x.new_empty(1, max_batch_size, x.size(2), requires_grad=False).bernoulli_(1 - self.dropout)
		x = x.masked_fill(m == 0, 0) / (1 - self.dropout)

		if is_packed:
			return PackedSequence(x, batch_sizes)
		else:
			return x

class BetterLSTM(torch.nn.LSTM):
	def __init__(self, *args, dropout_inp: float=0.2,
				 dropout_mid: float=0.2, dropout_out: float=0.2,  bidirectional=True,
				 batch_first=True, unit_forget_bias=True, **kwargs):
		super().__init__(*args, **kwargs,  bidirectional=bidirectional,  batch_first=batch_first)
		logging.warning("Better lstm initialized with dropout_inp : {}  dropout_mid {} dropout_out {} ".format(dropout_inp, dropout_mid, dropout_out))
		self.unit_forget_bias = unit_forget_bias
		self.dropout_mid = dropout_mid
		self.input_drop = VariationalDropout(dropout_inp, batch_first=batch_first)
		self.output_drop = VariationalDropout(dropout_out, batch_first=batch_first)
		self._init_weights()

	def _init_weights(self):
		"""
		Use orthogonal init for recurrent layers, xavier uniform for input layers
		Bias is 0 except for forget gate
		"""
		for name, param in self.named_parameters():
			if "weight_hh" in name:
				torch.nn.init.orthogonal_(param.data)
			elif "weight_ih" in name:
				torch.nn.init.xavier_uniform_(param.data)
			elif "bias" in name and self.unit_forget_bias:
				torch.nn.init.zeros_(param.data)
				param.data[self.hidden_size:2 * self.hidden_size] = 1

	def _drop_weights(self):
		for name, param in self.named_parameters():
			if "weight_hh" in name:
				getattr(self, name).data = torch.nn.functional.dropout(param.data, p=self.dropout_mid, training=self.training).contiguous()

	def forward(self, inp, hx=None):
		self._drop_weights()
		inp = self.input_drop(inp)
		seq, state = super().forward(inp, hx=hx )
		out_x = self.output_drop(seq)
		(h_0, c_0) = state
		# out_x, (h_0, c_0) 
		# logging.warning("  >>>>>> Better LSTM  out_x: {}  shape {} h_0 {} h_0.shape {} ".format( type(out_x), out_x.shape, type(h_0), h_0.shape ))
		return out_x, (h_0, c_0)


##############################################################################################################################################################
##############################################################################################################################################################




#-------------------- Adversarial Network ------------------------------------
# Brij: Added the gradient reversal layer






def to_cuda(m, x):
	"""Function to send tensor into corresponding device
	:param torch.nn.Module m: torch module
	:param torch.Tensor x: torch tensor
	:return: torch tensor located in the same place as torch module
	:rtype: torch.Tensor
	"""
	assert isinstance(m, torch.nn.Module)
	device = next(m.parameters()).device
	return x.to(device)



def th_accuracy(pad_outputs, pad_targets, ignore_label):
	"""Function to calculate accuracy
	:param torch.Tensor pad_outputs: prediction tensors (B*Lmax, D)
	:param torch.Tensor pad_targets: target tensors (B, Lmax, D)
	:param int ignore_label: ignore label id
	:retrun: accuracy value (0.0 - 1.0)
	:rtype: float
	"""
	# logging.warning(" pad_out_shape {} pad_targets_shape {}  ".format(pad_outputs.shape, pad_targets.shape))
	# pad_pred = pad_outputs.view(
	#     pad_targets.size(0),
	#     pad_targets.size(1),
	#     pad_outputs.size(1)).argmax(2)
	pad_pred = pad_outputs.argmax(1)
	mask = pad_targets != ignore_label
	# logging.warning("\n pad_pred: {}\npad_targ: {}\n".format(pad_pred[::500].tolist(), pad_targets[:500].tolist(), ))
	numerator = torch.sum(pad_pred.masked_select(mask) == pad_targets.masked_select(mask))
	denominator = torch.sum(mask)
	# logging.warning("******************************\n\n")
	return float(numerator) / float(denominator)

class ReverseLayerF(Function):

	@staticmethod
	def forward(ctx, x, alpha):
		ctx.alpha = alpha

		return x.view_as(x)

	@staticmethod
	def backward(ctx, grad_output):
		output = grad_output.neg() * ctx.alpha

		return output, None



# Brij: Added to classify speakers from encoder projections
class SpeakerAdv(torch.nn.Module):
	""" Speaker adversarial module
	:param int odim: dimension of outputs
	:param int eprojs: number of encoder projection units
	:param float dropout_rate: dropout rate (0.0 ~ 1.0)
	"""

	def __init__(self, odim, eprojs, advunits, advlayers, dropout_mid=0.0, dropout_inp=0.0, dropout_out=0.0):
		super(SpeakerAdv, self).__init__()
		device="cuda"
		margin=3
		self.advunits = advunits
		self.advlayers = advlayers

		# self.advnet = Xvector(lin_neurons=advunits, in_channels=eprojs)

		# self.advnet = TDNN(input_dim=eprojs, output_dim=advunits, context_size=9, dilation=2)
		# self.advnet2 = TDNN(input_dim=advunits, output_dim=advunits, context_size=9, dilation=1)
		# self.advnet3 = TDNN(input_dim=advunits, output_dim=advunits, context_size=7, dilation=2)
		# self.advnet4 = TDNN(input_dim=advunits, output_dim=advunits, context_size=7, dilation=1)
		# self.advnet5 = TDNN(input_dim=advunits, output_dim=advunits, context_size=5, dilation=2)
		# self.advnet6 = TDNN(input_dim=advunits, output_dim=advunits, context_size=5, dilation=1)
		# self.advnet5 = TDNN(input_dim=advunits, output_dim=advunits, context_size=3, dilation=2)
		# self.advnet6 = TDNN(input_dim=advunits, output_dim=advunits, context_size=3, dilation=1)

		# self.advnet = BetterLSTM(eprojs, advunits, self.advlayers, batch_first=True, dropout_mid=dropout_mid, dropout_inp=dropout_inp, dropout_out=dropout_out, bidirectional=True)
		self.advnet = torch.nn.LSTM(eprojs, advunits, self.advlayers, batch_first=True, dropout=dropout_mid, bidirectional=True)
		# self.advnet = DropoutLSTMModel( input_size=eprojs, hidden_size=advunits, n_layers=self.advlayers, dropoutw=dropout_rate, bidirectional=True)
		# logging.warning(" Created better lstm with dropout_w = {}  ".format(dropout_rate))
		self.output = torch.nn.Linear(2*advunits, odim)
		# self.lsoftmax_linear = LSoftmaxLinear(input_features=advunits, output_features=odim, margin=margin, device=device)
		# self.reset_parameters()


	def zero_state(self, hs_pad):
		return hs_pad.new_zeros(2*self.advlayers, hs_pad.size(0), self.advunits)


	def set_dropout(self, model, drop_rate=0):
		for name, child in model.named_children():
			if isinstance(child, torch.nn.Dropout):
				child.p = drop_rate
			self.set_dropout(child, drop_rate=drop_rate)


	def weight_reset(self, m):
		if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
			m.reset_parameters()





	def reinit_adv(self,):
		self.advnet.apply(self.weight_reset)
		self.output.reset_parameters()
		# self.output.apply(self.init_weights)

	def init_weights(self, m):
		if isinstance(m, torch.nn.Linear):
			torch.nn.init.xavier_uniform_(m.weight)
			m.bias.data.fill_(0.01)


	def forward(self, hs_pad, hlens, y_adv):
		'''Adversarial branch forward
		:param torch.Tensor hs_pad: batch of padded hidden state sequences (B, Tmax, D)
		:param torch.Tensor hlens: batch of lengths of hidden state sequences (B)
		:param torch.Tensor y_adv: batch of speaker class (B, #Speakers)
		:return: loss value
		:rtype: torch.Tensor
		:return: accuracy
		:rtype: float
		'''

		# initialization
		# logging.warning("initializing new modified forward")
		h_0 = self.zero_state(hs_pad)
		c_0 = self.zero_state(hs_pad)



		# for better lstm
		self.advnet.flatten_parameters()
		out_x, (h_0, c_0) = self.advnet(hs_pad, (h_0, c_0))


		# logging.warning(" ------>>> lstm hs_pad.shape {} h_0.shape {} c_0.shape {}  ".format(  hs_pad.shape, h_0.shape, c_0.shape) )

		#  for tudnn and xvector_tudnn
		# out_x = self.advnet(hs_pad)
		# out_x = self.advnet2(out_x)
		# out_x = self.advnet3(out_x)
		# out_x = self.advnet4(out_x)
		# out_x = self.advnet5(out_x)
		# out_x = self.advnet6(out_x)

		y_hat = self.output(out_x)

		# Create labels tensor by replicating speaker label
		batch_size, avg_seq_len, out_dim = y_hat.size()

		labels = torch.zeros([batch_size, avg_seq_len], dtype=torch.float64)
		for ix in range(batch_size):
			labels[ix, :] = y_adv[ix]

		# Mean over sequence length
		#y_hat = torch.mean(y_hat, 1)
		h_0.detach_()
		c_0.detach_()

		# Convert tensors to desired shape
		y_hat = y_hat.view((-1, out_dim))
		labels = labels.contiguous().view(-1)
		labels = to_cuda(self, labels.long())


		# logging.warning(" y_hat.shape {} labels.shape {} ".format(y_hat.shape, labels.shape))

		# in_features = 512
		# out_features =  10 # Number of classes

		# criterion = AngularPenaltySMLoss(in_features, out_features, loss_type='arcface') # loss_type in ['arcface', 'sphereface', 'cosface']
		# # Forward method works similarly to nn.CrossEntropyLoss
		# # x of shape (batch_size, in_features), labels of shape (batch_size,)
		# # labels should indicate class of each sample, and should be an int, l satisying 0 <= l < out_dim
		# loss = criterion(y_hat, labels)



		loss = F.cross_entropy(y_hat, labels, size_average=True)
		acc = th_accuracy(y_hat, labels, -1)

		return loss, acc







# # import torch
# # import torch.nn as nn
# from torch.autograd import Variable
# from torch.nn import Parameter
# # import torch.nn.functional as F


# class DropoutLSTMModel(torch.nn.Module):
#     def __init__(self, input_size, n_layers, hidden_size,
#                  dropout_i=0, dropout_h=0, return_states=True):
#         """
#         An LSTM model with Variational Dropout applied to the inputs and
#         model activations. For details see Eq. 7 of
#         A Theoretically Grounded Application of Dropout in Recurrent
#         Neural Networks. Gal & Ghahramani, 2016.
#         Note that this is equivalent to the weight-dropping scheme they
#         propose in Eq. 5 (but not Eq. 6).
#         Returns the hidden states for the final layer. Optionally also returns
#         the hidden and cell states for all layers.
#         Args:
#             input_size (int): input feature size.
#             n_layers (int): number of LSTM layers.
#             hidden_size (int): hidden layer size of all layers.
#             dropout_i (float): dropout rate of the inputs (t).
#             dropout_h (float): dropout rate of the state (t-1).
#             return_states (bool): If true, returns hidden and cell statees for
#                 all cells during the forward pass.
#         """
#         super(LSTMModel, self).__init__()

#         assert all([0 <= x < 1 for x in [dropout_i, dropout_h]])
#         assert all([0 < x for x in [input_size, n_layers, hidden_size]])
#         assert isinstance(return_states, bool)

#         self._input_size = input_size
#         self._n_layers = n_layers
#         self._hidden_size = hidden_size
#         self._dropout_i = dropout_i
#         self._dropout_h = dropout_h
#         self._return_states = return_states

#         cells = []
#         for i in range(n_layers):
#             cells.append(torch.nn.LSTMCell(input_size if i == 0 else hidden_size,
#                                      hidden_size,
#                                      bias=True))

#         self._cells = torch.nn.ModuleList(cells)
#         self._input_drop = SampleDrop(dropout=self._dropout_i)
#         self._state_drop = SampleDrop(dropout=self._dropout_h)

#     @property
#     def input_size(self):
#         return self._input_size

#     @property
#     def n_layers(self):
#         return self._n_layers

#     @property
#     def hidden_size(self):
#         return self._hidden_size

#     @property
#     def dropout_i(self):
#         return self._dropout_i

#     @property
#     def dropout_h(self):
#         return self._dropout_h

#     def _new_state(self, batch_size):
#         """Initalizes states."""
#         h = Variable(torch.zeros(batch_size, self._hidden_size))
#         c = Variable(torch.zeros(batch_size, self._hidden_size))

#         return (h, c)

#     def forward(self, X):
#         """Forward pass through the LSTM.
#         Args:
#             X (tensor): input with dimensions batch_size, seq_len, input_size
#         Returns: Output ht from the final LSTM cell, and optionally all
#             intermediate states.
#         """
#         states = [] if self._return_states else None
#         X = X.permute(1, 0, 2)
#         seq_len, batch_size, input_size = X.shape

#         for cell in self._cells:
#             ht, ct = [], []

#             # Initialize new state.
#             h, c = self._new_state(batch_size)
#             h = h.to(X.device)
#             c = c.to(X.device)

#             # Fix dropout weights for this cell.
#             self._input_drop.set_weights(X[0, ...])  # Removes time dimension.
#             self._state_drop.set_weights(h)

#             for sample in X:

#                 h, c = cell(self._input_drop(sample), (self._state_drop(h), c))
#                 ht.append(h)
#                 ct.append(c)

#             # Output is again [batch, seq_len, n_feat].
#             ht = torch.stack(ht, dim=0).permute(1, 0, 2)
#             ct = torch.stack(ct, dim=0).permute(1, 0, 2)

#             if self._return_states:
#                 states.append((ht, ct))

#             X = ht.clone().permute(1, 0, 2)  # Input for next cell.

#         return (ht, states)


# class SampleDrop(torch.nn.Module):
#     """Applies dropout to input samples with a fixed mask."""
#     def __init__(self, dropout=0):
#         super().__init__()

#         assert 0 <= dropout < 1
#         self._mask = None
#         self._dropout = dropout

#     def set_weights(self, X):
#         """Calculates a new dropout mask."""
#         assert len(X.shape) == 2

#         mask = Variable(torch.ones(X.size(0), X.size(1)), requires_grad=False)

#         if X.is_cuda:
#             mask = mask.cuda()

#         self._mask = F.dropout(mask, p=self._dropout, training=self.training)

#     def forward(self, X):
#         """Applies dropout to the input X."""
#         if not self.training or not self._dropout:
#             return X
#         else:
#             return X * self._mask




