"""Library implementing normalization.

Authors
 * Mirco Ravanelli 2020
 * Guillermo Cámbara 2021
 * Sarthak Yadav 2022
"""



import math
import torch
import logging
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from typing import Tuple

# import torch
import logging
# import torch.nn as nn

logger = logging.getLogger(__name__)



class BatchNorm1d(nn.Module):
    """Applies 1d batch normalization to the input tensor.

    Arguments
    ---------
    input_shape : tuple
        The expected shape of the input. Alternatively, use ``input_size``.
    input_size : int
        The expected size of the input. Alternatively, use ``input_shape``.
    eps : float
        This value is added to std deviation estimation to improve the numerical
        stability.
    momentum : float
        It is a value used for the running_mean and running_var computation.
    affine : bool
        When set to True, the affine parameters are learned.
    track_running_stats : bool
        When set to True, this module tracks the running mean and variance,
        and when set to False, this module does not track such statistics.
    combine_batch_time : bool
        When true, it combines batch an time axis.


    Example
    -------
    >>> input = torch.randn(100, 10)
    >>> norm = BatchNorm1d(input_shape=input.shape)
    >>> output = norm(input)
    >>> output.shape
    torch.Size([100, 10])
    """

    def __init__(
        self,
        input_shape=None,
        input_size=None,
        eps=1e-05,
        momentum=0.1,
        affine=True,
        track_running_stats=True,
        combine_batch_time=False,
        skip_transpose=False,
    ):
        super().__init__()
        self.combine_batch_time = combine_batch_time
        self.skip_transpose = skip_transpose

        if input_size is None and skip_transpose:
            input_size = input_shape[1]
        elif input_size is None:
            input_size = input_shape[-1]

        self.norm = nn.BatchNorm1d(
            input_size,
            eps=eps,
            momentum=momentum,
            affine=affine,
            track_running_stats=track_running_stats,
        )

    def forward(self, x):
        """Returns the normalized input tensor.

        Arguments
        ---------
        x : torch.Tensor (batch, time, [channels])
            input to normalize. 2d or 3d tensors are expected in input
            4d tensors can be used when combine_dims=True.
        """
        shape_or = x.shape
        if self.combine_batch_time:
            if x.ndim == 3:
                x = x.reshape(shape_or[0] * shape_or[1], shape_or[2])
            else:
                x = x.reshape(
                    shape_or[0] * shape_or[1], shape_or[3], shape_or[2]
                )

        elif not self.skip_transpose:
            x = x.transpose(-1, 1)

        x_n = self.norm(x)

        if self.combine_batch_time:
            x_n = x_n.reshape(shape_or)
        elif not self.skip_transpose:
            x_n = x_n.transpose(1, -1)

        return x_n


class BatchNorm2d(nn.Module):
    """Applies 2d batch normalization to the input tensor.

    Arguments
    ---------
    input_shape : tuple
        The expected shape of the input. Alternatively, use ``input_size``.
    input_size : int
        The expected size of the input. Alternatively, use ``input_shape``.
    eps : float
        This value is added to std deviation estimation to improve the numerical
        stability.
    momentum : float
        It is a value used for the running_mean and running_var computation.
    affine : bool
        When set to True, the affine parameters are learned.
    track_running_stats : bool
        When set to True, this module tracks the running mean and variance,
        and when set to False, this module does not track such statistics.

    Example
    -------
    >>> input = torch.randn(100, 10, 5, 20)
    >>> norm = BatchNorm2d(input_shape=input.shape)
    >>> output = norm(input)
    >>> output.shape
    torch.Size([100, 10, 5, 20])
    """

    def __init__(
        self,
        input_shape=None,
        input_size=None,
        eps=1e-05,
        momentum=0.1,
        affine=True,
        track_running_stats=True,
    ):
        super().__init__()

        if input_shape is None and input_size is None:
            raise ValueError("Expected input_shape or input_size as input")

        if input_size is None:
            input_size = input_shape[-1]

        self.norm = nn.BatchNorm2d(
            input_size,
            eps=eps,
            momentum=momentum,
            affine=affine,
            track_running_stats=track_running_stats,
        )

    def forward(self, x):
        """Returns the normalized input tensor.

        Arguments
        ---------
        x : torch.Tensor (batch, time, channel1, channel2)
            input to normalize. 4d tensors are expected.
        """
        x = x.transpose(-1, 1)
        x_n = self.norm(x)
        x_n = x_n.transpose(1, -1)

        return x_n


class LayerNorm(nn.Module):
    """Applies layer normalization to the input tensor.

    Arguments
    ---------
    input_shape : tuple
        The expected shape of the input.
    eps : float
        This value is added to std deviation estimation to improve the numerical
        stability.
    elementwise_affine : bool
        If True, this module has learnable per-element affine parameters
        initialized to ones (for weights) and zeros (for biases).

    Example
    -------
    >>> input = torch.randn(100, 101, 128)
    >>> norm = LayerNorm(input_shape=input.shape)
    >>> output = norm(input)
    >>> output.shape
    torch.Size([100, 101, 128])
    """

    def __init__(
        self,
        input_size=None,
        input_shape=None,
        eps=1e-05,
        elementwise_affine=True,
    ):
        super().__init__()
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if input_shape is not None:
            input_size = input_shape[2:]

        self.norm = torch.nn.LayerNorm(
            input_size,
            eps=self.eps,
            elementwise_affine=self.elementwise_affine,
        )

    def forward(self, x):
        """Returns the normalized input tensor.

        Arguments
        ---------
        x : torch.Tensor (batch, time, channels)
            input to normalize. 3d or 4d tensors are expected.
        """
        return self.norm(x)


class InstanceNorm1d(nn.Module):
    """Applies 1d instance normalization to the input tensor.

    Arguments
    ---------
    input_shape : tuple
        The expected shape of the input. Alternatively, use ``input_size``.
    input_size : int
        The expected size of the input. Alternatively, use ``input_shape``.
    eps : float
        This value is added to std deviation estimation to improve the numerical
        stability.
    momentum : float
        It is a value used for the running_mean and running_var computation.
    track_running_stats : bool
        When set to True, this module tracks the running mean and variance,
        and when set to False, this module does not track such statistics.
    affine : bool
        A boolean value that when set to True, this module has learnable
        affine parameters, initialized the same way as done for
        batch normalization. Default: False.

    Example
    -------
    >>> input = torch.randn(100, 10, 20)
    >>> norm = InstanceNorm1d(input_shape=input.shape)
    >>> output = norm(input)
    >>> output.shape
    torch.Size([100, 10, 20])
    """

    def __init__(
        self,
        input_shape=None,
        input_size=None,
        eps=1e-05,
        momentum=0.1,
        track_running_stats=True,
        affine=False,
    ):
        super().__init__()

        if input_shape is None and input_size is None:
            raise ValueError("Expected input_shape or input_size as input")

        if input_size is None:
            input_size = input_shape[-1]

        self.norm = nn.InstanceNorm1d(
            input_size,
            eps=eps,
            momentum=momentum,
            track_running_stats=track_running_stats,
            affine=affine,
        )

    def forward(self, x):
        """Returns the normalized input tensor.

        Arguments
        ---------
        x : torch.Tensor (batch, time, channels)
            input to normalize. 3d tensors are expected.
        """
        x = x.transpose(-1, 1)
        x_n = self.norm(x)
        x_n = x_n.transpose(1, -1)

        return x_n


class InstanceNorm2d(nn.Module):
    """Applies 2d instance normalization to the input tensor.

    Arguments
    ---------
    input_shape : tuple
        The expected shape of the input. Alternatively, use ``input_size``.
    input_size : int
        The expected size of the input. Alternatively, use ``input_shape``.
    eps : float
        This value is added to std deviation estimation to improve the numerical
        stability.
    momentum : float
        It is a value used for the running_mean and running_var computation.
    track_running_stats : bool
        When set to True, this module tracks the running mean and variance,
        and when set to False, this module does not track such statistics.
    affine : bool
        A boolean value that when set to True, this module has learnable
        affine parameters, initialized the same way as done for
        batch normalization. Default: False.

    Example
    -------
    >>> input = torch.randn(100, 10, 20, 2)
    >>> norm = InstanceNorm2d(input_shape=input.shape)
    >>> output = norm(input)
    >>> output.shape
    torch.Size([100, 10, 20, 2])
    """

    def __init__(
        self,
        input_shape=None,
        input_size=None,
        eps=1e-05,
        momentum=0.1,
        track_running_stats=True,
        affine=False,
    ):
        super().__init__()

        if input_shape is None and input_size is None:
            raise ValueError("Expected input_shape or input_size as input")

        if input_size is None:
            input_size = input_shape[-1]

        self.norm = nn.InstanceNorm2d(
            input_size,
            eps=eps,
            momentum=momentum,
            track_running_stats=track_running_stats,
            affine=affine,
        )

    def forward(self, x):
        """Returns the normalized input tensor.

        Arguments
        ---------
        x : torch.Tensor (batch, time, channel1, channel2)
            input to normalize. 4d tensors are expected.
        """
        x = x.transpose(-1, 1)
        x_n = self.norm(x)
        x_n = x_n.transpose(1, -1)

        return x_n


class GroupNorm(nn.Module):
    """Applies group normalization to the input tensor.

    Arguments
    ---------
    input_shape : tuple
        The expected shape of the input. Alternatively, use ``input_size``.
    input_size : int
        The expected size of the input. Alternatively, use ``input_shape``.
    num_groups : int
        Number of groups to separate the channels into.
    eps : float
        This value is added to std deviation estimation to improve the numerical
        stability.
    affine : bool
         A boolean value that when set to True, this module has learnable per-channel
         affine parameters initialized to ones (for weights) and zeros (for biases).
    Example
    -------
    >>> input = torch.randn(100, 101, 128)
    >>> norm = GroupNorm(input_size=128, num_groups=128)
    >>> output = norm(input)
    >>> output.shape
    torch.Size([100, 101, 128])
    """

    def __init__(
        self,
        input_shape=None,
        input_size=None,
        num_groups=None,
        eps=1e-05,
        affine=True,
    ):
        super().__init__()
        self.eps = eps
        self.affine = affine

        if input_shape is None and input_size is None:
            raise ValueError("Expected input_shape or input_size as input")

        if num_groups is None:
            raise ValueError("Expected num_groups as input")

        if input_shape is not None:
            input_size = input_shape[-1]

        self.norm = torch.nn.GroupNorm(
            num_groups, input_size, eps=self.eps, affine=self.affine,
        )

    def forward(self, x):
        """Returns the normalized input tensor.

        Arguments
        ---------
        x : torch.Tensor (batch, time, channels)
            input to normalize. 3d or 4d tensors are expected.
        """
        x = x.transpose(-1, 1)
        x_n = self.norm(x)
        x_n = x_n.transpose(1, -1)

        return x_n


class ExponentialMovingAverage(nn.Module):
    """
    Applies learnable exponential moving average, as required by learnable PCEN layer

    Arguments
    ---------
    input_size : int
        The expected size of the input.
    coeff_init: float
        Initial smoothing coefficient value
    per_channel: bool
        Controls whether every smoothing coefficients are learned
        independently for every input channel
    trainable: bool
        whether to learn the PCEN parameters or use fixed
    skip_transpose : bool
        If False, uses batch x time x channel convention of speechbrain.
        If True, uses batch x channel x time convention.

    Example
    -------
    >>> inp_tensor = torch.rand([10, 50, 40])
    >>> pcen = ExponentialMovingAverage(40)
    >>> out_tensor = pcen(inp_tensor)
    >>> out_tensor.shape
    torch.Size([10, 50, 40])
    """

    def __init__(
        self,
        input_size: int,
        coeff_init: float = 0.04,
        per_channel: bool = False,
        trainable: bool = True,
        skip_transpose: bool = False,
    ):
        super(ExponentialMovingAverage, self).__init__()
        self._coeff_init = coeff_init
        self._per_channel = per_channel
        self.skip_transpose = skip_transpose
        self.trainable = trainable
        weights = (
            torch.ones(input_size,) if self._per_channel else torch.ones(1,)
        )
        self._weights = nn.Parameter(
            weights * self._coeff_init, requires_grad=trainable
        )

    def forward(self, x):
        """Returns the normalized input tensor.

       Arguments
        ---------
        x : torch.Tensor (batch, time, channels)
            input to normalize.
        """
        if not self.skip_transpose:
            x = x.transpose(1, -1)
        w = torch.clamp(self._weights, min=0.0, max=1.0)
        initial_state = x[:, :, 0]

        def scan(init_state, x, w):
            """Loops and accumulates."""
            x = x.permute(2, 0, 1)
            acc = init_state
            results = []
            for ix in range(x.shape[0]):
                acc = (w * x[ix]) + ((1.0 - w) * acc)
                results.append(acc.unsqueeze(0))
            results = torch.cat(results, dim=0)
            results = results.permute(1, 2, 0)
            return results

        output = scan(initial_state, x, w)
        if not self.skip_transpose:
            output = output.transpose(1, -1)
        return output


class PCEN(nn.Module):
    """
    This class implements a learnable Per-channel energy normalization (PCEN) layer, supporting both
    original PCEN as specified in [1] as well as sPCEN as specified in [2]

    [1] Yuxuan Wang, Pascal Getreuer, Thad Hughes, Richard F. Lyon, Rif A. Saurous, "Trainable Frontend For
    Robust and Far-Field Keyword Spotting", in Proc of ICASSP 2017 (https://arxiv.org/abs/1607.05666)

    [2] Neil Zeghidour, Olivier Teboul, F{\'e}lix de Chaumont Quitry & Marco Tagliasacchi, "LEAF: A LEARNABLE FRONTEND
    FOR AUDIO CLASSIFICATION", in Proc of ICLR 2021 (https://arxiv.org/abs/2101.08596)

    The default argument values correspond with those used by [2].

    Arguments
    ---------
    input_size : int
        The expected size of the input.
    alpha: float
        specifies alpha coefficient for PCEN
    smooth_coef: float
        specified smooth coefficient for PCEN
    delta: float
        specifies delta coefficient for PCEN
    root: float
        specifies root coefficient for PCEN
    floor: float
        specifies floor coefficient for PCEN
    trainable: bool
        whether to learn the PCEN parameters or use fixed
    per_channel_smooth_coef: bool
        whether to learn independent smooth coefficients for every channel.
        when True, essentially using sPCEN from [2]
    skip_transpose : bool
        If False, uses batch x time x channel convention of speechbrain.
        If True, uses batch x channel x time convention.

    Example
    -------
    >>> inp_tensor = torch.rand([10, 50, 40])
    >>> pcen = PCEN(40, alpha=0.96)         # sPCEN
    >>> out_tensor = pcen(inp_tensor)
    >>> out_tensor.shape
    torch.Size([10, 50, 40])
    """

    def __init__(
        self,
        input_size,
        alpha: float = 0.96,
        smooth_coef: float = 0.04,
        delta: float = 2.0,
        root: float = 2.0,
        floor: float = 1e-12,
        trainable: bool = True,
        per_channel_smooth_coef: bool = True,
        skip_transpose: bool = False,
    ):
        super(PCEN, self).__init__()
        self._smooth_coef = smooth_coef
        self._floor = floor
        self._per_channel_smooth_coef = per_channel_smooth_coef
        self.skip_transpose = skip_transpose
        self.alpha = nn.Parameter(
            torch.ones(input_size) * alpha, requires_grad=trainable
        )
        self.delta = nn.Parameter(
            torch.ones(input_size) * delta, requires_grad=trainable
        )
        self.root = nn.Parameter(
            torch.ones(input_size) * root, requires_grad=trainable
        )

        self.ema = ExponentialMovingAverage(
            input_size,
            coeff_init=self._smooth_coef,
            per_channel=self._per_channel_smooth_coef,
            skip_transpose=True,
            trainable=trainable,
        )

    def forward(self, x):
        """Returns the normalized input tensor.

        Arguments
        ---------
        x : torch.Tensor (batch, time, channels)
            input to normalize.
        """
        if not self.skip_transpose:
            x = x.transpose(1, -1)
        alpha = torch.min(
            self.alpha, torch.tensor(1.0, dtype=x.dtype, device=x.device)
        )
        root = torch.max(
            self.root, torch.tensor(1.0, dtype=x.dtype, device=x.device)
        )
        ema_smoother = self.ema(x)
        one_over_root = 1.0 / root
        output = (
            x / (self._floor + ema_smoother) ** alpha.view(1, -1, 1)
            + self.delta.view(1, -1, 1)
        ) ** one_over_root.view(1, -1, 1) - self.delta.view(
            1, -1, 1
        ) ** one_over_root.view(
            1, -1, 1
        )
        if not self.skip_transpose:
            output = output.transpose(1, -1)
        return output

















"""Library implementing linear transformation.

Authors
 * Mirco Ravanelli 2020
 * Davide Borra 2021
"""



class Linear(torch.nn.Module):
    """Computes a linear transformation y = wx + b.

    Arguments
    ---------
    n_neurons : int
        It is the number of output neurons (i.e, the dimensionality of the
        output).
    input_shape: tuple
        It is the shape of the input tensor.
    input_size: int
        Size of the input tensor.
    bias : bool
        If True, the additive bias b is adopted.
    combine_dims : bool
        If True and the input is 4D, combine 3rd and 4th dimensions of input.

    Example
    -------
    >>> inputs = torch.rand(10, 50, 40)
    >>> lin_t = Linear(input_shape=(10, 50, 40), n_neurons=100)
    >>> output = lin_t(inputs)
    >>> output.shape
    torch.Size([10, 50, 100])
    """

    def __init__(
        self,
        n_neurons,
        input_shape=None,
        input_size=None,
        bias=True,
        combine_dims=False,
    ):
        super().__init__()
        self.combine_dims = combine_dims

        if input_shape is None and input_size is None:
            raise ValueError("Expected one of input_shape or input_size")

        if input_size is None:
            input_size = input_shape[-1]
            if len(input_shape) == 4 and self.combine_dims:
                input_size = input_shape[2] * input_shape[3]

        # Weights are initialized following pytorch approach
        self.w = nn.Linear(input_size, n_neurons, bias=bias)

    def forward(self, x):
        """Returns the linear transformation of input tensor.

        Arguments
        ---------
        x : torch.Tensor
            Input to transform linearly.
        """
        if x.ndim == 4 and self.combine_dims:
            x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3])

        wx = self.w(x)

        return wx


class LinearWithConstraint(Linear):
    """Computes a linear transformation y = wx + b with kernel max-norm constaint.
    This corresponds to set an upper bound for the kernel norm.

    Arguments
    ---------
    n_neurons : int
        It is the number of output neurons (i.e, the dimensionality of the
        output).
    input_shape: tuple
        It is the shape of the input tensor.
    input_size: int
        Size of the input tensor.
    bias : bool
        If True, the additive bias b is adopted.
    combine_dims : bool
        If True and the input is 4D, combine 3rd and 4th dimensions of input.
    max_norm : float
        Kernel max-norm

    Example
    -------
    >>> inputs = torch.rand(100,)
    >>> max_norm = 1.
    >>> lin_t_contrained = LinearWithConstraint(input_size=inputs.shape[0], n_neurons=2, max_norm=max_norm)
    >>> output = lin_t_contrained(inputs)
    >>> torch.any(torch.norm(lin_t_contrained.w.weight.data, p=2, dim=0)>max_norm)
    tensor(False)
    """

    def __init__(self, *args, max_norm=1, **kwargs):
        self.max_norm = max_norm
        super(LinearWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, x):
        """Returns the linear transformation of input tensor.

        Arguments
        ---------
        x : torch.Tensor
            Input to transform linearly.
        """
        self.w.weight.data = torch.renorm(
            self.w.weight.data, p=2, dim=0, maxnorm=self.max_norm
        )
        return super(LinearWithConstraint, self).forward(x)














class Conv1d(nn.Module):
    """This function implements 1d convolution.

    Arguments
    ---------
    out_channels : int
        It is the number of output channels.
    kernel_size : int
        Kernel size of the convolutional filters.
    input_shape : tuple
        The shape of the input. Alternatively use ``in_channels``.
    in_channels : int
        The number of input channels. Alternatively use ``input_shape``.
    stride : int
        Stride factor of the convolutional filters. When the stride factor > 1,
        a decimation in time is performed.
    dilation : int
        Dilation factor of the convolutional filters.
    padding : str
        (same, valid, causal). If "valid", no padding is performed.
        If "same" and stride is 1, output shape is the same as the input shape.
        "causal" results in causal (dilated) convolutions.
    groups: int
        Number of blocked connections from input channels to output channels.
    padding_mode : str
        This flag specifies the type of padding. See torch.nn documentation
        for more information.
    skip_transpose : bool
        If False, uses batch x time x channel convention of speechbrain.
        If True, uses batch x channel x time convention.
    weight_norm : bool
        If True, use weight normalization,
        to be removed with self.remove_weight_norm() at inference

    Example
    -------
    >>> inp_tensor = torch.rand([10, 40, 16])
    >>> cnn_1d = Conv1d(
    ...     input_shape=inp_tensor.shape, out_channels=8, kernel_size=5
    ... )
    >>> out_tensor = cnn_1d(inp_tensor)
    >>> out_tensor.shape
    torch.Size([10, 40, 8])
    """

    def __init__(
        self,
        out_channels,
        kernel_size,
        input_shape=None,
        in_channels=None,
        stride=1,
        dilation=1,
        padding="same",
        groups=1,
        bias=True,
        padding_mode="reflect",
        skip_transpose=False,
        weight_norm=False,
        conv_init=None,
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.padding = padding
        self.padding_mode = padding_mode
        self.unsqueeze = False
        self.skip_transpose = skip_transpose

        if input_shape is None and in_channels is None:
            raise ValueError("Must provide one of input_shape or in_channels")

        if in_channels is None:
            in_channels = self._check_input_shape(input_shape)

        self.in_channels = in_channels

        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            self.kernel_size,
            stride=self.stride,
            dilation=self.dilation,
            padding=0,
            groups=groups,
            bias=bias,
        )

        if conv_init == "kaiming":
            nn.init.kaiming_normal_(self.conv.weight)

        if weight_norm:
            self.conv = nn.utils.weight_norm(self.conv)

    def forward(self, x):
        """Returns the output of the convolution.

        Arguments
        ---------
        x : torch.Tensor (batch, time, channel)
            input to convolve. 2d or 4d tensors are expected.
        """
        if not self.skip_transpose:
            x = x.transpose(1, -1)

        if self.unsqueeze:
            x = x.unsqueeze(1)

        if self.padding == "same":
            x = self._manage_padding(
                x, self.kernel_size, self.dilation, self.stride
            )

        elif self.padding == "causal":
            num_pad = (self.kernel_size - 1) * self.dilation
            x = F.pad(x, (num_pad, 0))

        elif self.padding == "valid":
            pass

        else:
            raise ValueError(
                "Padding must be 'same', 'valid' or 'causal'. Got "
                + self.padding
            )

        wx = self.conv(x)

        if self.unsqueeze:
            wx = wx.squeeze(1)

        if not self.skip_transpose:
            wx = wx.transpose(1, -1)

        return wx

    def _manage_padding(
        self, x, kernel_size: int, dilation: int, stride: int,
    ):
        """This function performs zero-padding on the time axis
        such that their lengths is unchanged after the convolution.

        Arguments
        ---------
        x : torch.Tensor
            Input tensor.
        kernel_size : int
            Size of kernel.
        dilation : int
            Dilation used.
        stride : int
            Stride.
        """

        # Detecting input shape
        L_in = self.in_channels

        # Time padding
        padding = get_padding_elem(L_in, stride, kernel_size, dilation)

        # Applying padding
        x = F.pad(x, padding, mode=self.padding_mode)

        return x

    def _check_input_shape(self, shape):
        """Checks the input shape and returns the number of input channels."""

        if len(shape) == 2:
            self.unsqueeze = True
            in_channels = 1
        elif self.skip_transpose:
            in_channels = shape[1]
        elif len(shape) == 3:
            in_channels = shape[2]
        else:
            raise ValueError(
                "conv1d expects 2d, 3d inputs. Got " + str(len(shape))
            )

        # Kernel size must be odd
        if not self.padding == "valid" and self.kernel_size % 2 == 0:
            raise ValueError(
                "The field kernel size must be an odd number. Got %s."
                % (self.kernel_size)
            )

        return in_channels

    def remove_weight_norm(self):
        """Removes weight normalization at inference if used during training."""
        self.conv = nn.utils.remove_weight_norm(self.conv)












# https://github.com/speechbrain/speechbrain/blob/develop/speechbrain/nnet/CNN.py



    
def get_padding_elem(L_in: int, stride: int, kernel_size: int, dilation: int):
    """This function computes the number of elements to add for zero-padding.
    Arguments
    ---------
    L_in : int
    stride: int
    kernel_size : int
    dilation : int
    """
    if stride > 1:
        padding = [math.floor(kernel_size / 2), math.floor(kernel_size / 2)]

    else:
        L_out = (
            math.floor((L_in - dilation * (kernel_size - 1) - 1) / stride) + 1
        )
        padding = [
            math.floor((L_in - L_out) / 2),
            math.floor((L_in - L_out) / 2),
        ]
    return padding


def get_padding_elem_transposed(
    L_out: int,
    L_in: int,
    stride: int,
    kernel_size: int,
    dilation: int,
    output_padding: int,
):
    """This function computes the required padding size for transposed convolution
    Arguments
    ---------
    L_out : int
    L_in : int
    stride: int
    kernel_size : int
    dilation : int
    output_padding : int
    """

    padding = -0.5 * (
        L_out
        - (L_in - 1) * stride
        - dilation * (kernel_size - 1)
        - output_padding
        - 1
    )
    return int(padding)