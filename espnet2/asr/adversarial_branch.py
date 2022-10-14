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


    def reinit_adv(self,):
        self.advnet.reset_parameters()
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

        # logging.warning("Passing encoder output through advnet {} ".format(hs_pad.shape))
        # logging.warning(" spkid inside adversarial {} ".format(y_adv))
        self.advnet.flatten_parameters()
        out_x, (h_0, c_0) = self.advnet(hs_pad, (h_0, c_0))

        # logging.warning("advnet output size = %s", str(out_x.shape))
        # logging.warning("adversarial target size = %s", str(y_adv.shape))

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
        #labels = to_cuda(self, labels.float())
        # logging.warning("adversarial output size = %s", str(y_hat.shape))
        # logging.warning("artificial label size = %s", str(labels.shape))

        loss = F.cross_entropy(y_hat, labels, size_average=True)
        #loss = F.kl_div(y_hat, labels, size_average=True)
        # logging.warning("Adversarial loss = %f", loss.item())
        acc = th_accuracy(y_hat, labels, -1)
        # logging.warning("Adversarial accuracy = %f", acc)

        return loss, acc
    
    
    




#-------------------- Adversarial Network ------------------------------------


