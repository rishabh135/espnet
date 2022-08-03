from abc import ABC, abstractmethod

import argparse
import logging
from typing import Callable, Collection, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
# import warpctc_pytorch as warp_ctc
from torch.autograd import Function

from collections import Counter

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
    pad_pred = pad_outputs.view(
        pad_targets.size(0),
        pad_targets.size(1),
        pad_outputs.size(1)).argmax(2)
    mask = pad_targets != ignore_label
    # logging.warning("\n pad_pred: {}\npad_targ: {}\n".format(pad_pred[:500].tolist(), pad_targets[:500].tolist(), ))
    numerator = torch.sum(pad_pred.masked_select(mask) == pad_targets.masked_select(mask))
    denominator = torch.sum(mask)
    # logging.warning("******************************\n\n")
    return float(numerator) / float(denominator)*100

class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None






# class SpeakerAdv(torch.nn.Module):
#     """ Speaker adversarial module
#     :param int odim: dimension of outputs
#     :param int eprojs: number of encoder projection units
#     :param float dropout_rate: dropout rate (0.0 ~ 1.0)
#     """

#     def __init__(self, odim, eprojs, advunits, advlayers, dropout_rate=0.3):
#         super(SpeakerAdv, self).__init__()
#         self.advunits = advunits
#         self.advlayers = advlayers
#         self.odim_adv = odim
#         self.advnet = torch.nn.LSTM(eprojs, advunits, self.advlayers,
#                                     batch_first=True, dropout=dropout_rate,
#                                     bidirectional=True)
#         '''
#         linears = [torch.nn.Linear(eprojs, advunits), torch.nn.ReLU(),
#                    torch.nn.Dropout(p=dropout_rate)]
#         for l in six.moves.range(1, self.advlayers):
#             linears.extend([torch.nn.Linear(advunits, advunits),
#                             torch.nn.ReLU(), torch.nn.Dropout(p=dropout_rate)])
#         self.advnet = torch.nn.Sequential(*linears)
#         '''
#         #self.vgg = VGG2L(1)
#         #layer_arr = [torch.nn.Linear(get_vgg2l_odim(eprojs, in_channel=1),
#         #                                  advunits), torch.nn.ReLU()]
#         #for l in six.moves.range(1, self.advlayers):
#         #    layer_arr.extend([torch.nn.Linear(advunits, advunits),
#         #                    torch.nn.ReLU(), torch.nn.Dropout(p=dropout_rate)])
#         #self.advnet = torch.nn.Sequential(*layer_arr)

#         self.output = torch.nn.Linear(2*advunits, odim)

#     def zero_state(self, hs_pad):
#         return hs_pad.new_zeros(2*self.advlayers, hs_pad.size(0), self.advunits)

#     def init_like_chainer(self):
#         '''
#         Copied from E2E class so that adv branch can be separately
#         re-initialized
#         '''
#         """Initialize weight like chainer
#         chainer basically uses LeCun way: W ~ Normal(0, fan_in ** -0.5), b = 0
#         pytorch basically uses W, b ~ Uniform(-fan_in**-0.5, fan_in**-0.5)
#         however, there are two exceptions as far as I know.
#         - EmbedID.W ~ Normal(0, 1)
#         - LSTM.upward.b[forget_gate_range] = 1 (but not used in NStepLSTM)
#         """
#         def lecun_normal_init_parameters(module):
#             for p in module.parameters():
#                 data = p.data
#                 if data.dim() == 1:
#                     # bias
#                     data.zero_()
#                 elif data.dim() == 2:
#                     # linear weight
#                     n = data.size(1)
#                     stdv = 1. / math.sqrt(n)
#                     data.normal_(0, stdv)
#                 elif data.dim() == 4:
#                     # conv weight
#                     n = data.size(1)
#                     for k in data.size()[2:]:
#                         n *= k
#                     stdv = 1. / math.sqrt(n)
#                     data.normal_(0, stdv)
#                 else:
#                     raise NotImplementedError

#         def set_forget_bias_to_one(bias):
#             n = bias.size(0)
#             start, end = n // 4, n // 2
#             bias.data[start:end].fill_(1.)

#         lecun_normal_init_parameters(self)
#         for names in self.advnet._all_weights:
#             for name in filter(lambda n: "bias" in n,  names):
#                 bias = getattr(self.advnet, name)
#                 set_forget_bias_to_one(bias)

#     def forward(self, hs_pad, hlens, spkid):
#         '''Adversarial branch forward
#         :param torch.Tensor hs_pad: batch of padded hidden state sequences (B, Tmax, D)
#         :param torch.Tensor hlens: batch of lengths of hidden state sequences (B)
#         :param torch.Tensor y_adv: batch of speaker class (B, #Speakers)
#         :return: loss value
#         :rtype: torch.Tensor
#         :return: accuracy
#         :rtype: float
#         '''

#         # initialization
#         logging.warning("\n***************initializing hidden states for LSTM")
#         h_0 = self.zero_state(hs_pad)
#         c_0 = self.zero_state(hs_pad)

#         logging.warning("Passing encoder output through advnet %s",
#                      str(hs_pad.shape))

#         self.advnet.flatten_parameters()
#         out_x, (h_0, c_0) = self.advnet(hs_pad, (h_0, c_0))

#         logging.warning(" out_x advnet output size = %s", str(out_x.shape))
#         logging.warning(" y_adv adversarial target size = %s", str(spkid.shape))

#         y_hat = self.output(out_x)

#         # Create labels tensor by replicating speaker label
#         batch_size, avg_seq_len, out_dim = y_hat.size()


#         labels = torch.zeros([batch_size, avg_seq_len], dtype=torch.float64)
#         y_adv = spkid.repeat(1, avg_seq_len).view(-1, avg_seq_len)
#         for ix in range(batch_size):
#             labels[ix, :] = y_adv[ix]

#         # Mean over sequence length
#         #y_hat = torch.mean(y_hat, 1)
#         h_0.detach_()
#         c_0.detach_()

#         # Convert tensors to desired shape
#         y_hat = y_hat.view((-1, out_dim))
#         labels = labels.contiguous().view(-1)
#         labels = to_cuda(self, labels.long())
#         #labels = to_cuda(self, labels.float())
#         # logging.warning("y_hat {}  \n labels {} \n".format(y_hat, labels[:600]))
#         logging.warning("adversarial output size = %s", str(y_hat.shape))
#         logging.warning("artificial label size = %s", str(labels.shape))

    
#         distinct_list= labels.unique() 
#         logging.warning(" distinct_list : {} ".format(distinct_list))
#         loss = F.cross_entropy(y_hat, labels, size_average=False)
#         #loss = F.kl_div(y_hat, labels, size_average=True)
#         logging.warning("Adversarial loss = %f", loss.item())
#         acc = th_accuracy(y_hat, labels.unsqueeze(0), -1)
#         logging.warning("Adversarial accuracy = %f", acc)

#         logging.warning("*******************************\n\n")

#         return loss, acc



# Brij: Added to classify speakers from encoder projections
class SpeakerAdv(torch.nn.Module):
    """ Speaker adversarial module
    :param int odim: dimension of outputs
    :param int eprojs: number of encoder projection units
    :param float dropout_rate: dropout rate (0.0 ~ 1.0)
    """

    def __init__(self, odim, eprojs, advunits, advlayers, dropout_rate=0.2):
        super(SpeakerAdv, self).__init__()
        self.advunits = advunits
        self.advlayers = advlayers
        self.advnet = torch.nn.LSTM(eprojs, advunits, self.advlayers,
                                    batch_first=True, dropout=dropout_rate,
                                    bidirectional=True)
        '''
        linears = [torch.nn.Linear(eprojs, advunits), torch.nn.ReLU(),
                   torch.nn.Dropout(p=dropout_rate)]
        for l in six.moves.range(1, self.advlayers):
            linears.extend([torch.nn.Linear(advunits, advunits),
                            torch.nn.ReLU(), torch.nn.Dropout(p=dropout_rate)])
        self.advnet = torch.nn.Sequential(*linears)
        '''
        #self.vgg = VGG2L(1)
        #layer_arr = [torch.nn.Linear(get_vgg2l_odim(eprojs, in_channel=1),
        #                                  advunits), torch.nn.ReLU()]
        #for l in six.moves.range(1, self.advlayers):
        #    layer_arr.extend([torch.nn.Linear(advunits, advunits),
        #                    torch.nn.ReLU(), torch.nn.Dropout(p=dropout_rate)])
        #self.advnet = torch.nn.Sequential(*layer_arr)
        self.output = torch.nn.Linear(2*advunits, odim)


    def zero_state(self, hs_pad):
        return hs_pad.new_zeros(2*self.advlayers, hs_pad.size(0), self.advunits)

    def forward(self, hs_pad, hlens, spkid):
        '''Adversarial branch forward
        :param torch.Tensor hs_pad: batch of padded hidden state sequences (B, Tmax, D)
        :param torch.Tensor hlens: batch of lengths of hidden state sequences (B)
        :param torch.Tensor speech_length: list of speaker index so that we can create a per frame label to used for y_adv by extending the tensor to a dimnetion of (B, average_sequence_length) with each frame  (B)
        :param torch.Tensor y_adv: batch of speaker class (B, #Speakers)
        :return: loss value
        :rtype: torch.Tensor
        :return: accuracy
        :rtype: float
        '''
    
        # CUDA runtime error (59) : device-side assert triggered  
        # Necessary to replace speech_length to text_length
        # https://stackoverflow.com/questions/51691563/cuda-runtime-error-59-device-side-assert-triggered

       
        # initialization
        h_0 = self.zero_state(hs_pad)
        c_0 = self.zero_state(hs_pad)

        # logging.warning("Passing encoder output through advnet %s",str(hs_pad.shape))
        
       
        self.advnet.flatten_parameters()
        out_x, (h_0, c_0) = self.advnet(hs_pad, (h_0, c_0))

        # logging.warning("advnet output size = %s", str(out_x.shape))

        # print("advnet output size = {} \n".format(out_x.shape))
        
        # logging.warning("adversarial target size = %s", str(spkid.shape))
        
        # print("adversarial target size  = {} \n".format(y_adv.shape))
        
        y_hat = self.output(out_x)

        # Create labels tensor by replicating speaker label
        batch_size, avg_seq_len, out_dim = y_hat.size()

        # print(" y_hat size  = {} and batch size {}  \n".format(y_hat.shape, batch_size))
     
        # for every frame the same speaked index, hence extend the vector to mimic the save value across average sequence length
        labels = torch.zeros([batch_size, avg_seq_len], dtype=torch.int64)
        
        y_adv = spkid.repeat(1, avg_seq_len).view(-1, avg_seq_len)
        # logging.warning(" y_adv shape  = {} ".format(y_adv.shape))
        

        for ix in range(batch_size):
            labels[ix, :] = y_adv[ix]

        # logging.warning(" labels shape  = {} ".format(labels.shape))
        # Mean over sequence length
        #y_hat = torch.mean(y_hat, 1)
        h_0.detach_()
        c_0.detach_()

        # Convert tensors to desired shape
        y_hat = y_hat.view((-1, out_dim))
        labels = labels.contiguous().view(-1)
        labels = to_cuda(self, labels.long())
        # logging.warning("adversarial output size = %s", str(y_hat.shape))
        # logging.warning("artificial label size = %s", str(labels.shape))

        loss = F.cross_entropy(y_hat, labels, size_average=True)
        
        logging.warning("Adversarial loss = %f", loss.item())
        acc = th_accuracy(y_hat, labels.unsqueeze(0), -1)
        logging.warning("Adversarial accuracy = %f", acc)

        return loss, acc




#-------------------- Adversarial Network ------------------------------------


