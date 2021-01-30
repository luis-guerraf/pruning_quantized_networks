import numpy as np
import time
import torch
import torch.nn as nn


model_profiling_hooks = []
model_profiling_speed_hooks = []

name_space = 45
params_space = 15
macs_space = 15
seconds_space = 15

num_forwards = 10


class Timer(object):
    def __init__(self, verbose=False):
        self.verbose = verbose
        self.start = None
        self.end = None

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        self.time = self.end - self.start
        if self.verbose:
            print('Elapsed time: %f ms.' % self.time)


def get_params(self):
    """get number of params in module"""
    return np.sum(
        [np.prod(list(w.size())) for w in self.parameters()])


def run_forward(self, input):
    with Timer() as t:
        for _ in range(num_forwards):
            self.forward(*input)
            torch.cuda.synchronize()
    return int(t.time * 1e9 / num_forwards)


def conv_module_name_filter(name):
    """filter module name to have a short view"""
    filters = {
        'kernel_size': 'k',
        'stride': 's',
        'padding': 'pad',
        'bias': 'b',
        'groups': 'g',
    }
    for k in filters:
        name = name.replace(k, filters[k])
    return name


def module_profiling(self, input, output, verbose):
    ins = input[0].size()
    outs = output.size()
    # NOTE: There are some difference between type and isinstance, thus please be careful.
    t = type(self)
    if isinstance(self, nn.Conv2d):
        self.n_macs = (ins[1] * outs[1] *
                       self.kernel_size[0] * self.kernel_size[1] *
                       outs[2] * outs[3] // self.groups) * outs[0]
        # self.n_params = get_params(self)  # NOTE: This doesn't contemplate pruning
        self.n_params = ins[1] * outs[1] * self.kernel_size[0] * self.kernel_size[1]
        if hasattr(self, 'network'):
            if (self.network == 'BinaryConnect') or (self.network == 'BinaryNet'):
                self.n_bits = (self.n_params * 1)
            elif (self.network == 'XnorNet'):
                self.n_bits = (self.n_params * 1) + (outs[1] * 32)
            elif (self.network == 'DoReFa'):
                self.n_bits = (self.n_params * 2)
            elif (self.network == 'SemiReal') or (self.network == 'Real'):
                self.n_bits = (self.n_params * 32)
        else:
            self.n_bits = (self.n_params * 32)
        self.n_bits += (outs[1] * 32) if hasattr(self.bias, 'size') else 0
        self.n_params += outs[1] if hasattr(self.bias, 'size') else 0
        self.n_seconds = run_forward(self, input)
        self.name = conv_module_name_filter(self.__repr__())
    elif isinstance(self, nn.ConvTranspose2d):
        self.n_macs = (ins[1] * outs[1] *
                       self.kernel_size[0] * self.kernel_size[1] *
                       outs[2] * outs[3] // self.groups) * outs[0]
        self.n_params = get_params(self)
        self.n_seconds = run_forward(self, input)
        self.n_bits = (self.n_params * 1) + (self.bias.size(0) * 32) if \
                                hasattr(self.bias, 'size') else (self.n_params * 1)
        self.name = conv_module_name_filter(self.__repr__())
    elif isinstance(self, nn.Linear):
        self.n_macs = ins[1] * outs[1] * outs[0]
        # self.n_params = get_params(self)  # NOTE: This doesn't contemplate pruning
        self.n_params = ins[1] * outs[1]
        if hasattr(self, 'network'):
            if (self.network == 'BinaryConnect') or (self.network == 'BinaryNet'):
                self.n_bits = (self.n_params * 1)
            else:
                self.n_bits = (self.n_params * 32)
        else:
            self.n_bits = (self.n_params * 32)
        self.n_bits += (outs[1] * 32) if hasattr(self.bias, 'size') else 0
        self.n_params += outs[1] if hasattr(self.bias, 'size') else 0
        self.n_seconds = run_forward(self, input)
        self.name = self.__repr__()
    elif isinstance(self, nn.AvgPool2d):
        # NOTE: this function is correct only when stride == kernel size
        self.n_macs = ins[1] * ins[2] * ins[3] * ins[0]
        self.n_params = 0
        self.n_seconds = run_forward(self, input)
        self.n_bits = 0
        self.name = self.__repr__()
    elif isinstance(self, nn.AdaptiveAvgPool2d):
        # NOTE: this function is correct only when stride == kernel size
        self.n_macs = ins[1] * ins[2] * ins[3] * ins[0]
        self.n_params = 0
        self.n_seconds = run_forward(self, input)
        self.n_bits = 0
        self.name = self.__repr__()
    elif isinstance(self, nn.BatchNorm2d):
        self.n_macs = ins[1] * ins[2] * ins[3] * ins[0] * 2
        # self.n_params = get_params(self)   # NOTE: This doesn't contemplate pruning
        self.n_params = outs[1] * 2
        self.n_seconds = run_forward(self, input)
        self.n_bits = self.n_params * 32
        self.name = self.__repr__()
    elif isinstance(self, nn.BatchNorm1d):
        self.n_macs = ins[0] * ins[1] * 2
        # self.n_params = get_params(self)   # NOTE: This doesn't contemplate pruning
        self.n_params = outs[1] * 2
        self.n_seconds = run_forward(self, input)
        self.n_bits = self.n_params * 32
        self.name = self.__repr__()
    elif isinstance(self, nn.PReLU):
        self.n_macs = 0
        self.n_params = get_params(self)
        self.n_seconds = run_forward(self, input)
        self.n_bits = self.n_params * 32
        self.name = self.__repr__()
    else:
        # This works only in depth-first travel of modules.
        self.n_macs = 0
        self.n_params = 0
        self.n_seconds = 0
        self.n_bits = 0
        num_children = 0
        for m in self.children():
            self.n_macs += getattr(m, 'n_macs', 0)
            self.n_params += getattr(m, 'n_params', 0)
            self.n_seconds += getattr(m, 'n_seconds', 0)
            self.n_bits += getattr(m, 'n_bits', 0)
            num_children += 1
        ignore_zeros_t = [
            nn.Dropout2d, nn.Dropout, nn.Sequential,
            nn.ReLU6, nn.ReLU, nn.MaxPool2d,
            nn.modules.padding.ZeroPad2d, nn.modules.activation.Sigmoid,
        ]
        # if (not getattr(self, 'ignore_model_profiling', False) and
        #         self.n_macs == 0 and
        #         t not in ignore_zeros_t):
        #     print(
        #         'WARNING: leaf module {} has zero n_macs.'.format(type(self)))
        return
    if verbose:
        print(
            self.name.ljust(name_space, ' ') +
            '{:,}'.format(self.n_params).rjust(params_space, ' ') +
            '{:,}'.format(self.n_macs).rjust(macs_space, ' ') +
            '{:,}'.format(self.n_bits).rjust(seconds_space, ' ') +
            '{:,}'.format(self.n_seconds).rjust(seconds_space, ' '))
    return


def add_profiling_hooks(m, verbose):
    global model_profiling_hooks
    model_profiling_hooks.append(
      m.register_forward_hook(
        lambda m, input, output: module_profiling(
          m, input, output, verbose=verbose)))


def remove_profiling_hooks():
    global model_profiling_hooks
    for h in model_profiling_hooks:
        h.remove()
    model_profiling_hooks = []


def model_profiling(model, height, width, batch=1, channel=3, use_cuda=True,
                    verbose_minimal=True, verbose=False):
    """ Pytorch model profiling with input image size
    (batch, channel, height, width).
    The function exams the number of multiply-accumulates (n_macs).

    Args:
        model: pytorch model
        height: int
        width: int
        batch: int
        channel: int
        use_cuda: bool

    Returns:
        macs: int
        params: int

    """
    model.eval()
    data = torch.rand(batch, channel, height, width)
    device = torch.device("cuda:0" if use_cuda else "cpu")
    model = model.to(device)
    data = data.to(device)
    model.apply(lambda m: add_profiling_hooks(m, verbose=verbose))
    if verbose_minimal:
        print(
            'Item'.ljust(name_space, ' ') +
            'params'.rjust(macs_space, ' ') +
            'macs'.rjust(macs_space, ' ') +
            'bits'.rjust(seconds_space, ' ') +
            'nanosecs'.rjust(seconds_space, ' '))
    if verbose:
        print(''.center(name_space+params_space+macs_space+seconds_space, '-'))
    model(data)
    if verbose:
        print(''.center(name_space+params_space+macs_space+seconds_space, '-'))
    if verbose_minimal:
        print(
            'Total'.ljust(name_space, ' ') +
            '{:,}'.format(model.n_params).rjust(params_space, ' ') +
            '{:,}'.format(model.n_macs).rjust(macs_space, ' ') +
            '{:,}'.format(model.n_bits).rjust(seconds_space, ' ') +
            '{:,}'.format(model.n_seconds).rjust(seconds_space, ' '))
    remove_profiling_hooks()
    return model.n_macs, model.n_params, model.n_bits, model.n_seconds
