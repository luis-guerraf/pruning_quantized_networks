import torch
import pdb
import torch.nn as nn
import math
from torch.autograd import Variable
from torch.autograd import Function
import torch.nn.functional as F

# If the DoReFa bitwidhts are change, remember to change the profiling as well
DoReFa_bitwidths = {'W': 2, 'A': 2}

class Xnorize_W(Function):
    @staticmethod
    def forward(self, tensor):
        # Binarize
        n = tensor[0].nelement()
        s = tensor.size()
        m = tensor.norm(1, 3, keepdim=True).sum(2, keepdim=True).sum(1, keepdim=True).div(n)
        tensor = tensor.sign().mul(m.expand(s))

        return tensor

    @staticmethod
    def backward(self, grad_output):
        return grad_output


class Xnorize_A(Function):
    @staticmethod
    def forward(self, tensor, kernel_size, stride, padding):
        self.save_for_backward(tensor)
        size = kernel_size.item()
        stride = stride.item()
        padding = padding.item()

        # Compute matrix K like in paper (notation from paper)
        A = torch.mean(torch.abs(tensor), dim=1, keepdim=True)
        k = torch.cuda.FloatTensor(1, 1, size, size).fill_(1).div(size*size)
        K = nn.functional.conv2d(A, k, None, stride, padding)

        # Binarize
        tensor = tensor.sign()

        return tensor, K

    @staticmethod
    def backward(self, grad_output):
        input, = self.saved_tensors
        grad_output[input > 1] = 0
        grad_output[input < -1] = 0
        return grad_output, None, None, None


class Binarize_A(Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x.sign()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors

        # grad_input = grad_output.clone()
        # grad_input.masked_fill_(input>1.0, 0.0)
        # grad_input.masked_fill_(input<-1.0, 0.0)
        # mask_pos = (input>=0.0) & (input<1.0)
        # mask_neg = (input<0.0) & (input>=-1.0)
        # grad_input.masked_scatter_(mask_pos, input[mask_pos].mul_(-2.0).add_(2.0))
        # grad_input.masked_scatter_(mask_neg, input[mask_neg].mul_(2.0).add_(2.0))
        # return grad_input * grad_output

        grad_output[input > 1] = 0
        grad_output[input < -1] = 0
        return grad_output


class Binarize_W(Function):
    @staticmethod
    def forward(ctx, x):
        # ctx.save_for_backward(x)
        return x.sign()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


def DoReFa_A(x, numBits):
    # Assumed symmetric distribution of weights (i.e. range [-val, val])

    # Bring to range [0, 1] reducing impact of large values
    w_q = torch.tanh(x).div(2 * torch.max(torch.abs(torch.tanh(x)))) + 0.5

    # Quantize to k bits in range [0, 1]
    # w_q = w_q.mul(2 ** numBits - 1).round().div(2 ** numBits - 1)
    w_q = w_q.mul(2 ** numBits - 1)
    w_q = RoundNoGradient.apply(w_q)
    w_q = w_q.div(2 ** numBits - 1)

    # Affine to bring to range [-1, 1]
    w_q *= 2
    w_q -= 1

    return w_q


def DoReFa_W(x, numBits):
    # Assumed symmetric distribution of weights (i.e. range [-val, val])
    # Bring to range [0, 1] reducing impact of large values
    w_q = torch.tanh(x).div(2 * torch.max(torch.abs(torch.tanh(x)))) + 0.5

    # Quantize to k bits in range [0, 1]
    # w_q = w_q.mul(2 ** numBits - 1).round().div(2 ** numBits - 1)
    w_q = w_q.mul(2 ** numBits - 1)
    w_q = RoundNoGradient.apply(w_q)
    w_q = w_q.div(2 ** numBits - 1)

    # Affine to bring to range [-1, 1]
    w_q *= 2
    w_q -= 1

    return w_q


class RoundNoGradient(Function):
    @staticmethod
    def forward(ctx, x):
        return torch.round(x)

    @staticmethod
    def backward(ctx, g):
        return g


class PrunableBatchNorm2d(nn.BatchNorm2d):

    def __init__(self, *kargs, **kwargs):
        super(PrunableBatchNorm2d, self).__init__(*kargs, **kwargs)
        in_features = kargs[0]
        self.pruned = torch.zeros(in_features, dtype=torch.uint8)
        self.prune_or_zeroOut = 'zeroOut'

    def forward(self, input):
        if self.prune_or_zeroOut == 'prune':
            assert not self.training, "Training in 'prune' mode is incorrect because issue with batch norm"
            running_mean = self.running_mean[self.pruned != 1]
            running_var = self.running_var[self.pruned != 1]
            weight = self.weight[self.pruned != 1]
            bias = self.bias[self.pruned != 1]
        else:
            # Doesn't matter because next layer will zero out input
            # Note: It does matter with residual connections
            input.data = ZeroOutInput()(input.data, self.pruned)
            self.running_mean.data[self.pruned == 1] = 0
            self.running_var.data[self.pruned == 1] = 0
            self.weight.data[self.pruned == 1] = 0
            self.bias.data[self.pruned == 1] = 0

            running_mean = self.running_mean
            running_var = self.running_var
            weight = self.weight
            bias = self.bias

        ## Copied from original Batchnorm ##
        self._check_input_dim(input)

        exponential_average_factor = 0.0

        if self.training and self.track_running_stats:
            # TODO: if statement only here to tell the jit to skip emitting this when it is None
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        return F.batch_norm(
            input, running_mean, running_var, weight, bias,
            self.training or not self.track_running_stats,
            exponential_average_factor, self.eps)


class PrunableBatchNorm1d(nn.BatchNorm1d):

    def __init__(self, *kargs, **kwargs):
        super(PrunableBatchNorm1d, self).__init__(*kargs, **kwargs)
        in_features = kargs[0]
        self.pruned = torch.zeros(in_features, dtype=torch.uint8)
        self.prune_or_zeroOut = 'zeroOut'

    def forward(self, input):
        if self.prune_or_zeroOut == 'prune':
            running_mean = self.running_mean[self.pruned != 1]
            running_var = self.running_var[self.pruned != 1]
            weight = self.weight[self.pruned != 1]
            bias = self.bias[self.pruned != 1]
        else:
            # Doesn't matter because next layer will zero out input
            # Note: It does matter with residual connections
            input.data = ZeroOutInput()(input.data, self.pruned)
            self.running_mean.data[self.pruned == 1] = 0
            self.running_var.data[self.pruned == 1] = 0
            self.weight.data[self.pruned == 1] = 0
            self.bias.data[self.pruned == 1] = 0

            running_mean = self.running_mean
            running_var = self.running_var
            weight = self.weight
            bias = self.bias

        ## Copied from original Batchnorm ##
        self._check_input_dim(input)

        exponential_average_factor = 0.0

        if self.training and self.track_running_stats:
            # TODO: if statement only here to tell the jit to skip emitting this when it is None
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        return F.batch_norm(
            input, running_mean, running_var, weight, bias,
            self.training or not self.track_running_stats,
            exponential_average_factor, self.eps)


class BinarizeLinear(nn.Linear):

    def __init__(self, *kargs, **kwargs):
        super(BinarizeLinear, self).__init__(*kargs, **kwargs)

        in_features = kargs[0]
        out_features = kargs[1]
        self.network = 'BinaryNet'      # BinaryConnect, BinaryNet, XnorNet or Real
        self.pruned_input = torch.zeros(in_features, dtype=torch.uint8)
        self.pruned_output = torch.zeros(out_features, dtype=torch.uint8)
        self.prune_or_zeroOut = 'zeroOut'

    def forward(self, input):
        assert (self.network != 'XnorNet') and (self.network != 'DoReFa'), \
            "Not implemented for XnortNet and DoReFa. If implement, please change profiling accordingly"

        if not hasattr(self.weight, 'org'):
            self.weight.org = self.weight.data.clone()

        # Activation function (Not implemented yet for XnorNet and DoReFa)
        if (self.network == 'BinaryConnect') or (self.network == 'Real'):
            input = F.relu(input)
        elif self.network == 'SemiReal':
            input = F.hardtanh(input)
        else:
            input = Binarize_A.apply(input)

        # Weight Quantization function (Not implemented yet for XnorNet and DoReFa)
        if (self.network == 'SemiReal') or (self.network == 'Real'):
            self.weight.data.copy_(self.weight.org)
        else:
            self.weight.data = Binarize_W.apply(self.weight.org)

        # Prune layer
        # Note: Linear layer pruning only works with 1x1xc inputs
        if self.prune_or_zeroOut == 'prune':
            weight = self.weight[self.pruned_output != 1, :]
            weight = weight[:, self.pruned_input != 1]
        else:
            input.data = ZeroOutInput()(input.data, self.pruned_input)
            # Zeroing out the inputs is not enough because of residual connections
            weight = self.weight
            weight.data = ZeroOutWeights()(weight.data, self.pruned_input, self.pruned_output)

        out = nn.functional.linear(input, weight)

        if not self.bias is None:
            if self.prune_or_zeroOut == 'prune':
                bias = self.bias[self.pruned_output != 1]
            else:
                bias = self.bias
                bias.data = ZeroOutBias()(bias.data, self.pruned_output)

            out += bias.view(1, -1).expand_as(out)

        return out

    # Angle of the whole layer
    def angle(self):
        w = self.weight.org
        angle = torch.acos(w.abs().sum() / (torch.sqrt(w.pow(2).sum()) * math.sqrt(w.nelement()))) * 180 / math.pi
        return angle

    # Angles per neuron
    def angles_channel(self):
        w = self.weight.org
        angles = torch.acos(w.abs().sum(1) / (torch.sqrt(w.pow(2).sum(1)) * math.sqrt(w.shape[1]))) * 180 / math.pi
        return angles

    # Angles of kernels that convolve with previous filters
    def angles_prev_filters(self):
        w = self.weight.org
        angles = torch.acos(w.abs().sum(0) / (torch.sqrt(w.pow(2).sum(0)) * math.sqrt(w.shape[0]))) * 180 / math.pi
        return angles

    # Distances per neuron
    def distances_channel(self):
        w = self.weight.org
        distances = (1-torch.abs(w)).pow(2).sum(1)
        return distances

    # Distances of kernels that convolve with previous filters
    def distances_prev_filters(self):
        w = self.weight.org
        distances = (1-torch.abs(w)).pow(2).sum(0)
        return distances


class BinarizeConv2d(nn.Conv2d):

    def __init__(self, *kargs, **kwargs):
        super(BinarizeConv2d, self).__init__(*kargs, **kwargs)
        in_features = kargs[0]
        out_features = kargs[1]
        self.network = 'BinaryNet'      # BinaryConnect, BinaryNet, XnorNet, DoReFa, SemiReal or Real
        self.kernel_pruning = False
        self.pruned = torch.zeros(in_features * out_features, dtype=torch.uint8)    # For use with kernel pruning
        self.pruned_input = torch.zeros(in_features, dtype=torch.uint8)             # For use with channel pruning
        self.pruned_output = torch.zeros(out_features, dtype=torch.uint8)           # For use with channel pruning
        self.prune_or_zeroOut = 'zeroOut'

    def forward(self, input):
        if not hasattr(self.weight, 'org'):
            self.weight.org = self.weight.data.clone()

        # Activation function
        if (input.size(1) != 3):
            if (self.network == 'BinaryConnect') or (self.network == 'Real'):
                input = F.relu(input)
            # elif self.network == 'XnorNet':
            #     input.data, K = Xnorize_A.apply((input.data, Variable(torch.ByteTensor([self.kernel_size[0]])),
            #                       Variable(torch.ByteTensor([self.stride[0]])),
            #                       Variable(torch.ByteTensor([self.padding[0]]))))
            elif self.network == 'DoReFa':
                input = DoReFa_A(input, DoReFa_bitwidths['A'])
            elif self.network == 'SemiReal':
                input = F.hardtanh(input)
            else:
                input = Binarize_A.apply(input)

        # Weight Quantization function
        if (self.network == 'SemiReal') or (self.network == 'Real'):
            self.weight.data.copy_(self.weight.org)
        elif self.network == 'XnorNet':
            self.weight.data = Xnorize_W.apply(self.weight.org)
        elif self.network == 'DoReFa':
            self.weight.data = DoReFa_W(self.weight.org, DoReFa_bitwidths['W'])
        else:
            self.weight.data = Binarize_W.apply(self.weight.org)

        # Prune layer
        if self.prune_or_zeroOut == 'prune':
            weight = self.weight[self.pruned_output != 1, :, :, :]
            weight = weight[:, self.pruned_input != 1, :, :]
        else:
            input.data = ZeroOutInput()(input.data, self.pruned_input)
            # Zeroing out the inputs is not enough because of residual connections
            weight = self.weight
            weight.data = ZeroOutWeights()(weight.data, self.pruned_input, self.pruned_output)

        # Kernel pruning (only used for plots)
        if self.kernel_pruning:
            weight.data.view(-1, weight.shape[2], weight.shape[3])[self.pruned, :, :] = 0

        out = nn.functional.conv2d(input, weight, None, self.stride,
                                   self.padding, self.dilation, self.groups)

        # if (self.network == 'XnorNet') and (input.size(1) != 3):
        #     out *= K.expand_as(out)

        if not self.bias is None:
            if self.prune_or_zeroOut == 'prune':
                bias = self.bias[self.pruned_output != 1]
            else:
                bias = self.bias
                bias.data = ZeroOutBias()(bias.data, self.pruned_output)

            out += bias.view(1, -1, 1, 1).expand_as(out)

        return out

    # Angle of the whole layer
    def angle(self):
        w = self.weight.org
        angle = torch.acos(w.abs().sum() / (torch.sqrt(w.pow(2).sum()) * math.sqrt(w.nelement()))) * 180 / math.pi
        return angle

    # Distances per channel by averaging individual kernels
    def distances_channel(self):
        distances = self.distances_kernel().mean(1)
        return distances

    # Distances per channel by averaging whole channel (NOTE: Has not been adapted to all networks)
    def distances_channel_alternative(self):
        w = self.weight.org
        distances = (1-torch.abs(w)).pow(2).sum(3).sum(2).sum(1)
        return distances

    # Distances of kernels that convolve with previous filters
    def distances_prev_filters(self):
        distances = self.distances_kernel().mean(0)
        return distances

    # Distances per kernel
    def distances_kernel(self):
        w = self.weight.org.clone()
        if (self.network == 'BinaryConnect') or (self.network == 'BinaryNet'):
            w = Binarize_W.forward(None, w)
            distances = (w - self.weight.org).pow(2).sum(3).sum(2)
            # distances = (1 - torch.abs(w)).pow(2).sum(3).sum(2)
        elif self.network == 'XnorNet':
            w = Xnorize_W.forward(None, w)
            distances = (w - self.weight.org).pow(2).sum(3).sum(2)
        elif self.network == 'DoReFa':
            w = DoReFa_W(w, DoReFa_bitwidths['W'])
            distances = (w - self.weight.org).pow(2).sum(3).sum(2)

        return distances

    # Angles per kernel
    def angles_kernel(self):
        w = self.weight.org.clone()
        # Angle for XnorNet should be the same the BinaryNet
        if (self.network == 'BinaryConnect') or (self.network == 'BinaryNet') or (self.network == 'XnorNet'):
            angles = torch.acos(w.abs().sum(3).sum(2) / (torch.sqrt(w.pow(2).sum(3).sum(2)) * \
                                            math.sqrt(w.shape[2]*w.shape[3]))) * 180 / math.pi
        elif self.network == 'DoReFa':
            w = DoReFa_W(w, DoReFa_bitwidths['W'])
            angles = torch.zeros(w.shape[0], w.shape[1]).cuda()
            for out_channel in range(0, w.shape[0]):
                for in_channel in range(0, w.shape[1]):
                    kernel_r = self.weight.org[out_channel, in_channel, :, :].view(-1)
                    kernel_q = w[out_channel, in_channel, :, :].view(-1)
                    angles[out_channel, in_channel] = torch.acos(torch.dot(kernel_r, kernel_q) \
                                                        / (torch.norm(kernel_r, p=2)*torch.norm(kernel_q, p=2)) \
                                                                ) * 180 / math.pi

        return angles

    # Angles of kernels that convolve with previous filters
    def angles_prev_filters(self):
        angles = self.angles_kernel().mean(0)
        return angles

    # Angles per channel by averaging individual kernels
    def angles_channel(self):
        angles = self.angles_kernel().mean(1)
        return angles

    # Angles per channel  (NOTE: Has not been adapted to all networks)
    def angles_channel_alternative(self):
        w = self.weight.org
        angles = torch.acos(w.abs().sum(3).sum(2).sum(1) / (torch.sqrt(w.pow(2).sum(3).sum(2).sum(1)) * \
                                            math.sqrt(w.shape[1]*w.shape[2]*w.shape[3]))) * 180 / math.pi
        return angles


class PrunableLinear(nn.Linear):

    def __init__(self, *kargs, **kwargs):
        super(PrunableLinear, self).__init__(*kargs, **kwargs)

        in_features = kargs[0]
        out_features = kargs[1]
        self.pruned_input = torch.zeros(in_features, dtype=torch.uint8)
        self.pruned_output = torch.zeros(out_features, dtype=torch.uint8)
        self.prune_or_zeroOut = 'zeroOut'

    def forward(self, input):
        # Prune layer
        # Note: Linear layer pruning only works with 1x1xc inputs
        if self.prune_or_zeroOut == 'prune':
            weight = self.weight[self.pruned_output != 1, :]
            weight = weight[:, self.pruned_input != 1]
        else:
            input.data = ZeroOutInput()(input.data, self.pruned_input)
            # Zeroing out the inputs is not enough because of residual connections
            weight = self.weight
            weight.data = ZeroOutWeights()(weight.data, self.pruned_input, self.pruned_output)

        out = nn.functional.linear(input, weight)

        if not self.bias is None:
            if self.prune_or_zeroOut == 'prune':
                bias = self.bias[self.pruned_output != 1]
            else:
                bias = self.bias
                bias.data = ZeroOutBias()(bias.data, self.pruned_output)

            out += bias.view(1, -1).expand_as(out)

        return out


class PrunableConv2d(nn.Conv2d):

    def __init__(self, *kargs, **kwargs):
        super(PrunableConv2d, self).__init__(*kargs, **kwargs)
        in_features = kargs[0]
        out_features = kargs[1]
        self.pruned = torch.zeros(in_features * out_features, dtype=torch.uint8)    # For use with kernel pruning
        self.pruned_input = torch.zeros(in_features, dtype=torch.uint8)             # For use with channel pruning
        self.pruned_output = torch.zeros(out_features, dtype=torch.uint8)           # For use with channel pruning
        self.prune_or_zeroOut = 'zeroOut'

    def forward(self, input):
        # Prune layer
        if self.prune_or_zeroOut == 'prune':
            weight = self.weight[self.pruned_output != 1, :, :, :]
            weight = weight[:, self.pruned_input != 1, :, :]
        else:
            input.data = ZeroOutInput()(input.data, self.pruned_input)
            # Zeroing out the inputs is not enough because of residual connections
            weight = self.weight
            weight.data = ZeroOutWeights()(weight.data, self.pruned_input, self.pruned_output)

        out = nn.functional.conv2d(input, weight, None, self.stride,
                                   self.padding, self.dilation, self.groups)

        if not self.bias is None:
            if self.prune_or_zeroOut == 'prune':
                bias = self.bias[self.pruned_output != 1]
            else:
                bias = self.bias
                bias.data = ZeroOutBias()(bias.data, self.pruned_output)

            out += bias.view(1, -1, 1, 1).expand_as(out)

        return out


class ZeroOutInput(Function):
    def __init__(self):
        super(ZeroOutInput, self).__init__()

    def forward(self, input, prune):
        if input.dim() == 2:
            input[:, prune == 1] = 0
        if input.dim() == 4:
            input[:, prune == 1, :, :] = 0

        return input

    def backward(self, grad_output):
       return grad_output, None


class ZeroOutWeights(Function):
    def __init__(self):
        super(ZeroOutWeights, self).__init__()

    def forward(self, weight, pruned_input, pruned_output):
        if weight.dim() == 2:
            weight[pruned_output == 1, :] = 0
            weight[:, pruned_input == 1] = 0
        if weight.dim() == 4:
            weight[pruned_output == 1, :, :, :] = 0
            weight[:, pruned_input == 1, :, :] = 0

        return weight

    def backward(self, grad_output):
        return grad_output, None, None


class ZeroOutBias(Function):
    def __init__(self):
        super(ZeroOutBias, self).__init__()

    def forward(self, bias, pruned_output):
        bias[pruned_output == 1] = 0

        return bias

    def backward(self, grad_output):
        return grad_output, None

