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

from espnet2.asr.loss_functions import AngularPenaltySMLoss


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



		# https://github.com/cvqluu/Angular-Penalty-Softmax-Losses-Pytorch
		# in_features = eprojs
		# out_features = odim# Number of classes
		# self.criterion = AngularPenaltySMLoss(out_features, out_features, loss_type='arcface') # loss_type in ['arcface', 'sphereface', 'cosface']







		# self.advnet = Xvector(lin_neurons=advunits, in_channels=eprojs)
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



		#  for tudnn and xvector_tudnn
		# out_x = self.advnet(hs_pad)
		# out_x = self.advnet2(out_x)
		# out_x = self.advnet3(out_x)
		# out_x = self.advnet4(out_x)
		# out_x = self.advnet5(out_x)
		# out_x = self.advnet6(out_x)

		y_hat = self.output(out_x)
		# logging.warning(" ---->>> lstm hs_pad.shape {} h_0.shape {} c_0.shape {}  y_hat {} ".format(  hs_pad.shape, h_0.shape, c_0.shape, y_hat.shape) )


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

		# logging.warning(" >>>>> y_hat {}  labels {}  ".format(  y_hat.shape, labels.shape) )


		# criterion_loss = self.criterion(y_hat, labels)

		loss = F.cross_entropy(y_hat, labels, size_average=True)
		acc = th_accuracy(y_hat, labels, -1)

		return loss, acc





